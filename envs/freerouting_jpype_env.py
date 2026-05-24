import glob
import hashlib
import os
import warnings

import gym
import numpy as np

try:
    import jpype
except Exception:
    jpype = None


class FreeroutingJPypeEnv(gym.Env):
    """Freerouting environment with legacy and newer JPype backend support."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        jar_path="",
        dsn_file_path="",
        dsn_files_list=None,
        seed=0,
        action_size=100,
        obs_shape=(64, 64, 3),
        max_steps=200,
        intersection_penalty_scale=0.1,
    ):
        super().__init__()
        self._rng = np.random.RandomState(seed)
        self._max_steps = max_steps
        self._step_count = 0
        self._dummy_mode = True
        self._backend = "dummy"
        self._controller = None
        self._status_controller = None
        self._session_controller = None
        self._job_controller = None
        self._intersection_penalty_scale = intersection_penalty_scale

        # Handle multi-DSN setup
        self._dsn_files_list = dsn_files_list if dsn_files_list else []
        if dsn_file_path and not self._dsn_files_list:
            self._dsn_files_list = [dsn_file_path]
        self._current_dsn_path = dsn_file_path or (
            self._dsn_files_list[0] if self._dsn_files_list else ""
        )
        self._dsn_file_path = self._current_dsn_path  # For backward compatibility

        self.action_space = gym.spaces.Discrete(action_size)
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=obs_shape,
                    dtype=np.uint8,
                )
            }
        )

        print("Freerouting backend: initializing")

        if not jpype:
            warnings.warn("JPype not installed; using dummy freerouting environment.")
            print("Freerouting backend: dummy")
            return

        # Validate jar path and DSN files
        if not jar_path or not os.path.exists(jar_path):
            warnings.warn(
                "Freerouting jar path is missing/invalid; using dummy freerouting environment."
            )
            print("Freerouting backend: dummy")
            return

        if not self._dsn_files_list:
            warnings.warn("No DSN files provided; using dummy freerouting environment.")
            print("Freerouting backend: dummy")
            return

        # Validate all DSN files exist
        for dsn_file in self._dsn_files_list:
            if not os.path.exists(dsn_file):
                warnings.warn(
                    f"DSN file not found: {dsn_file}; using dummy freerouting environment."
                )
                print("Freerouting backend: dummy")
                return

        try:
            if not jpype.isJVMStarted():
                jpype.startJVM(classpath=[jar_path])
        except Exception as err:
            warnings.warn(
                f"Failed to start JVM for freerouting ({err}); using dummy environment."
            )
            print("Freerouting backend: dummy")
            return

        try:
            if self._class_exists("app.freerouting.logic.RoutingController"):
                self._init_legacy_backend()
            elif self._class_exists("app.freerouting.api.v1.SystemControllerV1"):
                self._init_v1_backend()
            else:
                warnings.warn(
                    "No known freerouting JPype backend class found in jar; using dummy environment."
                )
                print("Freerouting backend: dummy")
        except Exception as err:
            warnings.warn(
                f"Failed to initialize freerouting JPype backend ({err}); using dummy environment."
            )
            print("Freerouting backend: dummy")

    def _class_exists(self, class_name):
        try:
            jpype.JClass(class_name)
            return True
        except Exception as err:
            if "UnsupportedClassVersionError" in str(err):
                warnings.warn(
                    "Freerouting jar needs a newer Java runtime. "
                    "Install a newer JDK and rerun to enable JPype backend."
                )
            return False

    def _init_legacy_backend(self):
        routing_controller = jpype.JClass("app.freerouting.logic.RoutingController")
        self._controller = routing_controller(self._current_dsn_path)
        self._dummy_mode = False
        self._backend = "jpype_legacy"
        print(
            f"Freerouting backend: JPype legacy RoutingController (DSN: {self._current_dsn_path})"
        )

    def _init_v1_backend(self):
        self._status_controller = jpype.JClass(
            "app.freerouting.api.v1.SystemControllerV1"
        )()
        self._session_controller = jpype.JClass(
            "app.freerouting.api.v1.SessionControllerV1"
        )()
        self._job_controller = jpype.JClass("app.freerouting.api.v1.JobControllerV1")()
        _ = self._response_status(self._status_controller.getStatus())
        self._dummy_mode = False
        self._backend = "jpype_api_v1"
        print("Freerouting backend: JPype API v1 controllers")

    def _dummy_obs(self):
        image = self._rng.randint(
            0,
            256,
            size=self.observation_space["image"].shape,
            dtype=np.uint8,
        )
        return {"image": image}

    def _text_obs(self, text):
        digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
        seed = int.from_bytes(digest[:4], byteorder="little", signed=False)
        rng = np.random.RandomState(seed)
        image = rng.randint(
            0,
            256,
            size=self.observation_space["image"].shape,
            dtype=np.uint8,
        )
        return {"image": image}

    def _response_status(self, response):
        try:
            return int(response.getStatus())
        except Exception:
            return -1

    def _response_entity_text(self, response):
        try:
            entity = response.getEntity()
            if entity is None:
                return str(response)
            return str(entity)
        except Exception:
            return str(response)

    def load_dsn_file(self, dsn_file_path):
        """Load a new DSN file dynamically. Works for legacy backend only."""
        if self._dummy_mode or not self._controller:
            return False

        try:
            if self._backend == "jpype_legacy":
                # For legacy backend, create a new controller with the new DSN file
                routing_controller = jpype.JClass(
                    "app.freerouting.logic.RoutingController"
                )
                self._controller = routing_controller(dsn_file_path)
                self._current_dsn_path = dsn_file_path
                self._dsn_file_path = dsn_file_path
                return True
            elif self._backend == "jpype_api_v1":
                # API v1 backend doesn't support DSN switching in the same session
                # Would require session reset/reload which is complex
                return False
        except Exception as err:
            warnings.warn(f"Failed to load DSN file {dsn_file_path}: {err}")
            return False

        return False

    def _detect_intersections(self):
        """Detect trace intersections in the current routing.
        Returns a penalty value based on intersection count."""
        if self._dummy_mode or not self._controller:
            return 0.0

        try:
            if self._backend == "jpype_legacy":
                # Check if there's a method to get unrouted nets or violations
                # The intersection penalty can be derived from unrouted nets count
                # Higher unrouted count = more potential intersections
                if hasattr(self._controller, "getUnroutedNets"):
                    unrouted_count = self._controller.getUnroutedNets()
                    if unrouted_count > 0:
                        # Penalty based on unrouted nets (proxy for intersections)
                        penalty = self._intersection_penalty_scale * min(
                            unrouted_count, 10
                        )
                        return penalty

                # Alternative: check for violations/conflicts if available
                if hasattr(self._controller, "getViolationCount"):
                    violation_count = self._controller.getViolationCount()
                    penalty = self._intersection_penalty_scale * violation_count
                    return penalty

            elif self._backend == "jpype_api_v1":
                # For API v1, check status code for violations
                status_code = self._response_status(self._status_controller.getStatus())
                if status_code >= 500:
                    return self._intersection_penalty_scale
        except Exception as err:
            warnings.warn(f"Error detecting intersections: {err}")

        return 0.0

    def reset(self):
        self._step_count = 0

        # Sample a random DSN file if multiple are available
        if len(self._dsn_files_list) > 1:
            sampled_dsn = self._rng.choice(self._dsn_files_list)
            if sampled_dsn != self._current_dsn_path:
                self.load_dsn_file(sampled_dsn)

        if self._dummy_mode:
            obs = self._dummy_obs()
            obs["is_first"] = True
            obs["is_terminal"] = False
            return obs

        if self._backend == "jpype_legacy":
            self._controller.reset()
            java_observation = self._controller.getObservation()
            image = np.asarray(java_observation, dtype=np.uint8)
            return {"image": image, "is_first": True, "is_terminal": False}

        if self._backend == "jpype_api_v1":
            try:
                health = self._status_controller.getStatus()
                obs = self._text_obs(self._response_entity_text(health))
            except Exception as err:
                warnings.warn(
                    f"API v1 reset failed ({err}); switching to dummy observations."
                )
                self._dummy_mode = True
                self._backend = "dummy"
                obs = self._dummy_obs()
            obs["is_first"] = True
            obs["is_terminal"] = False
            return obs

        obs = self._dummy_obs()
        obs["is_first"] = True
        obs["is_terminal"] = False
        return obs

    def step(self, action):
        self._step_count += 1

        if self._dummy_mode:
            obs = self._dummy_obs()
            obs["is_first"] = False
            done = self._step_count >= self._max_steps
            obs["is_terminal"] = bool(done)
            return obs, 0.0, done, {"backend": "dummy"}

        if self._backend == "jpype_legacy":
            reward = float(self._controller.performAction(int(action)))
            # Apply intersection penalty
            intersection_penalty = self._detect_intersections()
            reward -= intersection_penalty

            java_observation = self._controller.getObservation()
            image = np.asarray(java_observation, dtype=np.uint8)
            done = bool(self._controller.isFinished())
            return (
                {
                    "image": image,
                    "is_first": False,
                    "is_terminal": done,
                },
                reward,
                done,
                {"backend": self._backend, "dsn_file": self._current_dsn_path},
            )

        if self._backend == "jpype_api_v1":
            done = self._step_count >= self._max_steps
            reward = 0.0
            try:
                import sys
                import time

                t0 = time.time()
                health = self._status_controller.getStatus()
                t1 = time.time()
                if t1 - t0 > 0.1:
                    print(
                        f"[SLOW] getStatus took {t1-t0:.3f}s",
                        flush=True,
                        file=sys.stderr,
                    )
                status_code = self._response_status(health)
                text = self._response_entity_text(health)
                obs = self._text_obs(f"{status_code}:{text}:{int(action)}")
                if status_code >= 500:
                    reward = -1.0

                # Apply intersection penalty on top of any other penalties
                intersection_penalty = self._detect_intersections()
                reward -= intersection_penalty

                if self._step_count % 100 == 0:
                    print(
                        f"[V1] step {self._step_count} ok", flush=True, file=sys.stderr
                    )
            except Exception as err:
                warnings.warn(
                    f"API v1 step failed ({err}); falling back to dummy mode."
                )
                self._dummy_mode = True
                self._backend = "dummy"
                obs = self._dummy_obs()

            obs["is_first"] = False
            obs["is_terminal"] = bool(done)
            return (
                obs,
                reward,
                done,
                {"backend": "jpype_api_v1", "dsn_file": self._current_dsn_path},
            )

        obs = self._dummy_obs()
        obs["is_first"] = False
        done = self._step_count >= self._max_steps
        obs["is_terminal"] = bool(done)
        return obs, 0.0, done, {"backend": "dummy"}

    def close(self):
        return None
