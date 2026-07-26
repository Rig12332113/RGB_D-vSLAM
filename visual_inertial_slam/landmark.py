import cv2
import numpy as np
from scipy.sparse import lil_matrix

class Landmark:
    ID: int
    mean: np.ndarray
    covariance: np.ndarray
    descriptor: np.ndarray
    last_seen_frame: int

    def __init__(self, 
                 id, 
                 mean, 
                 covariance, 
                 descriptor,
                 seen_frame,
                 ):
        self.ID = id
        self.mean = mean
        self.covariance = covariance
        self.descriptor = descriptor
        self.last_seen_frame = seen_frame
        

class LandmarkMap:
    landmarks: dict[int, Landmark]
    maximum_count: int 
    next_id: int
    current_frame: int

    def __init__(self, maximum_point=300):
        self.landmarks = {}
        self.maximum_point = maximum_point
        self.next_id = 0
        self.current_frame = 0
        self.landmark_ids = []
        self.mu = np.empty((0, 1))
        self.sigma = np.empty((0, 0))
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def create_landmark(self, 
                        mean: np.ndarray,
                        covariance: np.ndarray,
                        descriptor: np.ndarray,
                        seen_frame: int):
        
        if len(self.landmarks) >= self.maximum_point:
            return None
        landmark = Landmark(self.next_id, mean, covariance, descriptor, seen_frame)
        self.next_id += 1
        self.landmarks[landmark.ID] = landmark

        self.landmark_ids.append(landmark.ID)
        self.mu = np.vstack((
            self.mu,
            mean.reshape(3, 1),
        ))

        old_size = self.sigma.shape[0]
        new_sigma = np.zeros((old_size + 3, old_size + 3))
        new_sigma[:old_size, :old_size] = self.sigma
        new_sigma[old_size:, old_size:] = covariance.reshape(3, 3)
        self.sigma = new_sigma

        return landmark

    def match_landmarks(
        self,
        descriptors: np.ndarray,
        ratio_threshold: float = 0.75,
        maximum_distance: float = 50.0,
    ) -> tuple[list[tuple[int, int, float]], list[int]]:
        """Match current ORB descriptors to active map landmarks.

        Returns:
            matches: (observation_index, landmark_id, Hamming distance)
            unmatched: observation indices without an accepted match
        """
        if descriptors is None:
            return [], []

        descriptors = np.asarray(descriptors, dtype=np.uint8)
        if descriptors.size == 0:
            return [], []
        if descriptors.ndim == 1:
            descriptors = descriptors.reshape(1, -1)
        if descriptors.ndim != 2 or descriptors.shape[1] != 32:
            raise ValueError("ORB descriptors must have shape (N, 32)")

        active_landmarks = [
            landmark
            for landmark in self.landmarks.values()
            if landmark.descriptor is not None
        ]
        if not active_landmarks:
            return [], list(range(len(descriptors)))

        landmark_ids = [landmark.ID for landmark in active_landmarks]
        map_descriptors = np.asarray(
            [landmark.descriptor for landmark in active_landmarks],
            dtype=np.uint8,
        ).reshape(-1, 32)

        neighbor_count = min(2, len(map_descriptors))
        nearest_neighbors = self.matcher.knnMatch(
            descriptors,
            map_descriptors,
            k=neighbor_count,
        )

        candidates = []
        for observation_index, neighbors in enumerate(nearest_neighbors):
            if not neighbors:
                continue

            best = neighbors[0]
            if best.distance > maximum_distance:
                continue

            if len(neighbors) == 2:
                second_best = neighbors[1]
                if best.distance >= ratio_threshold * second_best.distance:
                    continue

            candidates.append(
                (
                    observation_index,
                    landmark_ids[best.trainIdx],
                    float(best.distance),
                )
            )

        # Resolve competition for a landmark by accepting the lowest-distance
        # candidate first.
        candidates.sort(key=lambda match: match[2])
        matches = []
        used_observations = set()
        used_landmarks = set()

        for match in candidates:
            observation_index, landmark_id, _ = match
            if observation_index in used_observations:
                continue
            if landmark_id in used_landmarks:
                continue

            matches.append(match)
            used_observations.add(observation_index)
            used_landmarks.add(landmark_id)

        unmatched = [
            index
            for index in range(len(descriptors))
            if index not in used_observations
        ]
        matches.sort(key=lambda match: match[0])
        return matches, unmatched

    def discard_stale_landmarks(
        self,
        matches: list[tuple[int, int, float]],
        maximum_age: int = 5,
    ) -> list[np.ndarray]:
        """Discard stale landmarks and return their last world-frame means."""
        matched_ids = {landmark_id for _, landmark_id, _ in matches}

        stale_landmark_ids = [
            landmark.ID
            for landmark in self.landmarks.values()
            if landmark.ID not in matched_ids
            and self.current_frame - landmark.last_seen_frame > maximum_age
        ]
        discarded_points_world = [
            self.landmarks[landmark_id].mean.reshape(3).copy()
            for landmark_id in stale_landmark_ids
        ]

        landmark_indices = {
            landmark_id: index
            for index, landmark_id in enumerate(self.landmark_ids)
        }
        state_indices = []
        for landmark_id in stale_landmark_ids:
            landmark_index = landmark_indices[landmark_id]
            state_indices.extend(range(3 * landmark_index, 3 * landmark_index + 3))

        if state_indices:
            self.mu = np.delete(self.mu, state_indices, axis=0)
            self.sigma = np.delete(self.sigma, state_indices, axis=0)
            self.sigma = np.delete(self.sigma, state_indices, axis=1)

        for landmark_id in stale_landmark_ids:
            self.landmark_ids.remove(landmark_id)
            del self.landmarks[landmark_id]

        return discarded_points_world
    
    def update_landmarks(self, 
                         matches,
                         points_optic,
                         T_OW):

        if not matches:
            return 
        
        H = lil_matrix((3 * len(matches), 3 * len(self.landmark_ids)))
        diff = np.zeros((3 * len(matches),  1))

        id_to_index = {
            landmark_id: index
            for index, landmark_id in enumerate(self.landmark_ids)
        }

        for observation_number, match in enumerate(matches):
            observation_idx, landmark_id, _ = match

            landmark_idx = id_to_index[landmark_id]

            row = 3 * observation_number
            col = 3 * landmark_idx

            z = points_optic[observation_idx].reshape(3, 1)
            landmark_world = self.mu[col:col+3]

            z_hat = T_OW[:3,:3] @ landmark_world + T_OW[:3, 3:4]
            diff[row:row+3] = z - z_hat
            H[row:row+3, col:col+3] = T_OW[:3,:3]

        H = H.tocsr()
        H_sigma = H @ self.sigma
        measurement_covariance = np.eye(3 * len(matches)) * 0.0025
        S = H @ H_sigma.T + measurement_covariance
        K = np.linalg.solve(
            S,
            H_sigma,
        ).T
        self.mu += K @ diff
        self.sigma -= K @ H_sigma
        self.sigma = 0.5 * (self.sigma + self.sigma.T)

        for _, landmark_id, _ in matches:
            self.landmarks[landmark_id].last_seen_frame = self.current_frame

        for index, landmark_id in enumerate(self.landmark_ids):
            start = 3 * index
            end = start + 3

            landmark = self.landmarks[landmark_id]
            landmark.mean = self.mu[start:end].copy()
            landmark.covariance = self.sigma[start:end, start:end].copy()
