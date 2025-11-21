# Import necessary libraries
import pandas as pd  # For data manipulation and reading CSV files
import matplotlib.pyplot as plt  # For plotting images and points
from pyproj import Transformer  # For coordinate system transformations (Geodetic <-> ECEF)
from PIL import Image  # For opening and handling image files
import pymap3d as pm  # For coordinate system transformations (ECEF <-> ENU)
import numpy as np  # For numerical operations, especially with arrays
import cv2  # OpenCV library for computer vision tasks like camera calibration and projection
import os  # For interacting with the operating system, like joining paths
import requests
from io import BytesIO

# added
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.patheffects as patheffects
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor


def process_data(data):
    def make_spin_gif(fig, ax, gif_path, fps=30, seconds=10, dpi=96):
        def update(angle):
            ax.view_init(elev=20, azim=angle)
            return ax,

        frames = np.linspace(0, 360, fps * seconds)

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=1000 * seconds / len(frames),
            blit=False,
        )

        anim.save(gif_path, writer="pillow", fps=fps, dpi=dpi)
        compress_gif(gif_path)


    def compress_gif(file):
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", file,
                "-vf", "split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
                "temp.gif",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        os.replace("temp.gif", file)


    # === Setup Local coordinate system ===

    os.makedirs(data["out"], exist_ok = True)

    # Store the origin GPS coordinates in a tuple
    origin_gps = (data["origin"]["latitude"], data["origin"]["longitude"], data["origin"]["altitude"])


    # === Load the Points CSV ===


    # Read the CSV file into a pandas DataFrame
    # This file should contain columns for point ID, image coordinates (img_x, img_y),
    # and corresponding map coordinates (map_lat, map_lng, map_altitude)
    poi_data = pd.read_csv(data["path_to_csv"])

    # Display the first few rows of the DataFrame to verify it loaded  correctly
    print(poi_data.head())


    # === Let's plot some points ===


    # Let's plot the x,y points we just read in on top of the image and make sure everything makes sense.

    # Grab the x,y points we just read in from the CSV above and put it into a numpy array:
    poi_xy = poi_data[['img_x', 'img_y']].to_numpy() # Shape: (N, 2)

    # Now grab the image:
    # If you have a local file, you can put the path here.
    # default_img = Image.open("local path...")

    # more complicated version to find it from a url:
    response = requests.get(data["url"])
    default_img = Image.open(BytesIO(response.content))

    # Now create the figure:

    # Create a figure for plotting
    plt.figure(figsize=(10, 8)) # Adjust figure size as needed

    # Display the background image
    plt.imshow(default_img)

    # Plot the original image points (from the input CSV) in red crosses
    plt.scatter(poi_xy[:, 0], poi_xy[:, 1], c='red', s=80, marker='x', label='Original POIs (Input Data)')
    # plt.show()


    # === Utitlity Functions for Projection and Calibration ===


    # Define transformation functions for coordinate systems.
    # GPS is (lat, long, alt)
    # ECEF is coordinates centered at the center of the earth
    # ENU is local east nort up relative to a local origin

    def gps_to_ecef(points_gps):
        """
        Converts an array of GPS points (latitude, longitude, altitude) to
        Earth-Centered, Earth-Fixed (ECEF) coordinates.

        Args:
            points_gps (np.ndarray): An Nx3 array where each row is [latitude, longitude, altitude].

        Returns:
            np.ndarray: An Nx3 array where each row is [ECEF_X, ECEF_Y, ECEF_Z].
        """
        # Define the transformation from Geodetic (EPSG:4979) to ECEF (EPSG:4978)
        # always_xy=True ensures the transformer expects (longitude, latitude) order
        transformer_geodetic_to_ecef = Transformer.from_crs(
            "epsg:4979", "epsg:4978", always_xy=True)

        # Perform the transformation
        # Note: pyproj expects longitude, latitude, altitude order for transform
        eX, eY, eZ = transformer_geodetic_to_ecef.transform(
            points_gps[:, 1], points_gps[:, 0], points_gps[:, 2])

        # Stack the results into an Nx3 NumPy array
        ecef_points = np.vstack((eX, eY, eZ)).T
        return ecef_points

    def ecef_to_enu(origin_gps, points_ecef):
        """
        Converts ECEF coordinates to local East-North-Up (ENU) coordinates
        relative to a specified ECEF origin point.

        Args:
            origin_gps (np.ndarray): A 1x3 array representing the origin in GPS [latitude, longitude, altitude].
            points_ecef (np.ndarray): An Nx3 array of points in ECEF [X, Y, Z].

        Returns:
            np.ndarray: An Nx3 array of points in ENU [East, North, Up] relative to the origin.
        """
        lat0, lon0, alt0 = origin_gps

        # Use pymap3d to convert the ECEF points to ENU relative to the Geodetic origin
        east, north, up = pm.ecef2enu(points_ecef[:, 0], points_ecef[:, 1], points_ecef[:, 2],
                                      lat0, lon0, alt0)

        # Stack the results into an Nx3 NumPy array and transpose to get the correct shape
        return np.vstack((east, north, up)).T

    def estimate_camera_params(poi_enu, poi_xy, frame_size, intrinsics_estimate=None, distortion_estimate=None, should_print = True):
        """
        Estimates camera intrinsic (K) and extrinsic (R, T) parameters using OpenCV's calibrateCamera.

        This function takes known 3D points in ENU coordinates and their corresponding 2D projections
        in the image to estimate the camera's properties.

        Args:
            poi_enu (np.ndarray): Nx3 array of Points of Interest (POIs) in ENU coordinates.
            poi_xy (np.ndarray): Nx2 array of corresponding image coordinates (pixels) for the POIs.
            frame_size (tuple): The (height, width) of the camera image in pixels.
            intrinsics_estimate (np.ndarray, optional): A 3x3 initial guess for the intrinsic matrix (K).
                                                        If None, a default guess based on frame size is used.
            distortion_estimate (np.ndarray, optional): A 5x1 initial guess for distortion coefficients.
                                                         If None, zeros are used.

        Returns:
            tuple: (camera_matrix, dist_coeffs, R, T)
                camera_matrix (np.ndarray): The estimated 3x3 intrinsic matrix (K).
                dist_coeffs (np.ndarray): The estimated 5x1 distortion coefficients.
                R (np.ndarray): The estimated 3x3 rotation matrix (world/ENU to camera).
                T (np.ndarray): The estimated 3x1 translation vector (world/ENU to camera).
        """

        # --- Step 1: Prepare Data for OpenCV ---
        # Ensure arrays are contiguous and have the correct types for OpenCV functions.
        object_points = np.ascontiguousarray(poi_enu, dtype=np.float32) # (N, 3) - 3D points in ENU
        image_points = np.ascontiguousarray(poi_xy, dtype=np.float32)   # (N, 2) - 2D projections

        # --- Step 2: Set Initial Guesses and Parameters for Calibration ---
        # If no initial intrinsic guess is provided, create a basic one.
        # Assumes focal length is large and principal point is image center.
        if intrinsics_estimate is None:
            # A large initial focal length guess can sometimes help convergence.
            estimated_focal_dist = max(frame_size) * 1.5 # Example heuristic
            intrinsics_estimate = np.array([
                [estimated_focal_dist, 0, frame_size[1] / 2], # fx, 0, cx
                [0, estimated_focal_dist, frame_size[0] / 2], # 0, fy, cy
                [0, 0, 1]
            ], dtype=np.float32)

        # If no initial distortion guess is provided, assume no distortion.
        if distortion_estimate is None:
            distortion_estimate = np.zeros((5, 1), dtype=np.float32)

        # Define termination criteria for the iterative optimization process.
        # Use stricter criteria than default for potentially better accuracy.
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            10000,    # Max iterations
            1e-12     # Epsilon (convergence tolerance)
        )

        if should_print:
            # Print initial guesses for debugging/information
            print("Initial intrinsics guess:\n", intrinsics_estimate)
            print("Initial distortion guess:\n", distortion_estimate.ravel())

        # Define flags to control the calibration process.
        # CALIB_USE_INTRINSIC_GUESS: Start optimization from the provided intrinsic estimate.
        # CALIB_FIX_PRINCIPAL_POINT: Keep cx, cy fixed. Often reasonable if the center is well-known.
        # CALIB_FIX_ASPECT_RATIO: Keep fx/fy ratio fixed. Common for modern cameras.
        # CALIB_ZERO_TANGENT_DIST: Assume tangential distortion (p1, p2) is zero. Often a safe assumption.
        # CALIB_FIX_K[4,5,6]: Do not estimate higher-order radial distortion coefficients.
        calibrate_flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS |
            cv2.CALIB_FIX_PRINCIPAL_POINT |
            cv2.CALIB_FIX_ASPECT_RATIO |
            cv2.CALIB_ZERO_TANGENT_DIST |
            cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
            cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
            # Note: CALIB_USE_EXTRINSIC_GUESS is removed as we don't provide R, T guesses here.
        )

        # --- Step 3: Perform Camera Calibration ---
        # calibrateCamera requires objectPoints and imagePoints to be lists of arrays (one per view).
        # Since we only have one view, we wrap them in lists.
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            [object_points],      # List containing the Nx3 object points array
            [image_points],       # List containing the Nx2 image points array
            (frame_size[1], frame_size[0]), # frame_size expected as (width, height) by OpenCV
            intrinsics_estimate,  # Initial intrinsic guess
            distortion_estimate,  # Initial distortion guess
            flags=calibrate_flags,
            criteria=criteria
        )

        # --- Step 6: Extract and Format Results ---
        # calibrateCamera returns lists of rvecs and tvecs (one per view). Get the first element.
        t_vector = tvecs[0]
        # Convert the returned rotation vector (Rodrigues) back to a 3x3 rotation matrix.
        r_matrix, _ = cv2.Rodrigues(rvecs[0])

        if should_print:
            # Check the return value (ret) which indicates success/quality (reprojection error)
            print(f"Calibration reprojection error: {ret}")

        # Return the estimated parameters and the camera origin in ECEF
        return camera_matrix, dist_coeffs, r_matrix, t_vector

    # added
    def camera_center_enu(r_matrix, t_vector):
        return (-r_matrix.T @ t_vector).ravel()

    def gps_to_camxy(points_gps, cam_k, cam_r, cam_t, camera_gps_origin, distortion=None):
        """
        Projects GPS points (latitude, longitude, altitude) onto the camera's image plane.

        This function converts GPS coordinates to ECEF, then to ENU relative to the camera origin,
        then transforms ENU points to the camera's coordinate system, and finally projects
        them onto the 2D image plane using camera intrinsics, extrinsics, and distortion coefficients.

        Args:
            points_gps (np.ndarray): An Nx3 array of GPS points [latitude, longitude, altitude].
            cam_k (np.ndarray): 3x3 camera intrinsic matrix (K).
            cam_r (np.ndarray): 3x3 camera rotation matrix (R) transforming ENU to Camera coords.
            cam_t (np.ndarray): 3x1 camera translation vector (T) transforming ENU to Camera coords.
            camera_gps_origin (tuple): The (latitude, longitude, altitude) of the camera origin.
            distortion (np.ndarray, optional): Camera distortion coefficients (k1, k2, p1, p2, k3). Defaults to None (no distortion).

        Returns:
            tuple: (image_x, image_y, cam_distance)
                image_x (np.ndarray): Array of projected x-coordinates in the image (pixels). NaN if point is behind camera.
                image_y (np.ndarray): Array of projected y-coordinates in the image (pixels). NaN if point is behind camera.
                cam_distance (np.ndarray): Array of distances (Z-coordinate) of points in the camera coordinate system.
        """


        # --- Step 1: Convert Input GPS points to ECEF ---
        # Convert all GPS points to ECEF using the dedicated function
        ecef_points = gps_to_ecef(points_gps) # Shape: (N, 3)
        num_points = ecef_points.shape[0]

        # --- Step 2: Convert ECEF points to ENU relative to Camera Origin ---
        # Convert ECEF points to local ENU coordinates using the camera's ECEF position as the origin
        enu_points = ecef_to_enu(camera_gps_origin, ecef_points) # Shape: (N, 3)

        # --- Step 3: Transform ENU points to Camera Coordinate System ---
        # Apply rotation (R) and translation (T) to transform ENU points to the camera's coordinate system.
        # Camera coordinates: X=right, Y=down, Z=forward (into scene)
        # Formula: P_cam = R * P_enu + T
        # We transpose enu_points for matrix multiplication and then transpose back.
        points_cam = (cam_r @ enu_points.T + cam_t).T # Shape: (N, 3)

        # --- Step 4: Handle Points Behind the Camera ---
        # Identify points with Z <= 0, which are behind the camera's image plane.
        behind_camera_mask = points_cam[:, 2] <= 0

        # --- Step 5: Project Points onto Image Plane using OpenCV ---
        # Initialize output arrays with NaN. NaN will remain for points behind the camera.
        image_x = np.full(num_points, np.nan)
        image_y = np.full(num_points, np.nan)
        # The Z coordinate in the camera system represents the distance along the camera's optical axis.
        cam_distance = points_cam[:, 2]

        # Only project points that are in front of the camera (Z > 0)
        valid_points_mask = ~behind_camera_mask
        if np.any(valid_points_mask):
            # Select only the valid ENU points for projection
            enu_points_to_project = enu_points[valid_points_mask]

            # OpenCV's projectPoints function requires:
            # - Object points (in the world/ENU frame for solvePnP/calibrateCamera, but here R and T already map ENU to Cam)
            #   However, projectPoints applies R and T itself. It expects world points, R, T that map world to camera.
            #   Since our R and T map ENU (our 'world' relative to camera origin) to Camera, we provide ENU points.
            # - Rotation vector (rvec): Rodrigues representation of cam_r (world-to-camera rotation)
            # - Translation vector (tvec): cam_t (world-to-camera translation)
            # - Camera matrix (cam_k)
            # - Distortion coefficients (distortion)

            # Convert the 3x3 rotation matrix to a rotation vector (Rodrigues format)
            rvec, _ = cv2.Rodrigues(cam_r)
            # Ensure tvec is float32
            tvec = cam_t.astype(np.float32)

            # If no distortion coefficients are provided, use zeros
            if distortion is None:
                distortion = np.zeros((5, 1), dtype=np.float32)

            # Project the valid 3D points (in ENU) onto the 2D image plane
            image_points, _ = cv2.projectPoints(
                enu_points_to_project.astype(np.float32), # Object points (Nx3)
                rvec,                                      # Rotation vector (3x1)
                tvec,                                      # Translation vector (3x1)
                cam_k,                                     # Camera matrix (3x3)
                distortion                                 # Distortion coefficients (5x1)
            )
            # Reshape the output image points from (N, 1, 2) to (N, 2)
            image_points = image_points.reshape(-1, 2)

            # Assign the calculated image coordinates back to the corresponding original indices
            valid_indices = np.where(valid_points_mask)[0]
            image_x[valid_indices] = image_points[:, 0]
            image_y[valid_indices] = image_points[:, 1]

        # Return the projected x, y coordinates and the camera-frame distances (Z)
        return image_x, image_y, cam_distance


    def calculate_fov_from_intrinsics(intrinsics, image_width, image_height):
        """
        Calculate horizontal and vertical field of view (FOV) in degrees
        from the camera intrinsic matrix (K).

        **NOTE:** This calculation assumes a simple pinhole camera model without lens distortion.
                 The FOV might differ if significant distortion is present.

        Args:
            intrinsics (np.ndarray): 3x3 camera intrinsic matrix (K).
            image_width (int): Width of the image in pixels.
            image_height (int): Height of the image in pixels.

        Returns:
            tuple: (hfov_deg, vfov_deg)
                hfov_deg (float): Horizontal field of view in degrees.
                vfov_deg (float): Vertical field of view in degrees.
        """
        # Extract focal lengths (fx, fy) and principal point (cx, cy) from the intrinsic matrix
        fx = intrinsics[0, 0]  # Focal length in x-direction (pixels)
        fy = intrinsics[1, 1]  # Focal length in y-direction (pixels)
        cx = intrinsics[0, 2]  # Principal point x-coordinate (pixels)
        cy = intrinsics[1, 2]  # Principal point y-coordinate (pixels)

        # Calculate horizontal FOV
        # Angle from optical center to the left edge + angle from optical center to the right edge
        # tan(angle_left) = cx / fx
        angle_left = np.arctan(cx / fx)
        # tan(angle_right) = (image_width - cx) / fx
        angle_right = np.arctan((image_width - cx) / fx)
        hfov_rad = angle_left + angle_right # Total horizontal FOV in radians

        # Calculate vertical FOV
        # Angle from optical center to the top edge + angle from optical center to the bottom edge
        # tan(angle_top) = cy / fy
        angle_top = np.arctan(cy / fy)
        # tan(angle_bottom) = (image_height - cy) / fy
        angle_bottom = np.arctan((image_height - cy) / fy)
        vfov_rad = angle_top + angle_bottom # Total vertical FOV in radians

        # Convert radians to degrees
        hfov_deg = np.degrees(hfov_rad)
        vfov_deg = np.degrees(vfov_rad)

        # Return FOV in degrees
        return (hfov_deg, vfov_deg)


    # === Initial Camera Intrinsic Matrix ===


    # --- Initial Camera Intrinsic Matrix (K) Guess ---

    # Get image dimensions (height, width)
    frame_size = (default_img.height, default_img.width) # OpenCV uses (height, width)

    full_frame_width_mm = 36.0
    ff_scale = full_frame_width_mm / frame_size[1]  # mm per "fx pixel" in 35mm-equivalent

    # Calculate an initial guess for the focal length (in pixels) based on the vertical FOV estimate.
    # Assumes the principal point is roughly centered (cy = height / 2).
    # Formula derived from pinhole model: tan(vfov / 2) = (height / 2) / fy
    # Rearranging gives: fy = (height / 2) / tan(vfov / 2)
    focal_length_estimate = 0.5 * frame_size[0] / np.tan(0.5 * np.radians(data["vfov_deg_estimate"]))

    # Create the initial guess for the 3x3 intrinsic matrix K.
    # Assumes square pixels (fx = fy) and principal point at the image center.
    initial_k = np.array([
        [focal_length_estimate, 0, frame_size[1] / 2], # [fx, 0, cx]
        [0, focal_length_estimate, frame_size[0] / 2], # [0, fy, cy]
        [0, 0, 1]
    ], dtype=np.float32)


    # === Prepare Data for Calibration ===


    # Extract 2D image coordinates (poi_xy) from the DataFrame
    poi_xy = poi_data[['img_x', 'img_y']].to_numpy() # Shape: (N, 2)

    # Extract 3D GPS coordinates (poi_gps) from the DataFrame
    poi_gps = poi_data[['map_lat', 'map_lng', 'map_altitude']].to_numpy() # Shape: (N, 3)

    # Convert the 3D GPS coordinates to ECEF coordinates
    poi_ecef = gps_to_ecef(poi_gps) # Shape: (N, 3)

    origin_ecef = gps_to_ecef(np.array([origin_gps])) # Shape: (1, 3)

    poi_enu = ecef_to_enu(origin_gps, poi_ecef)

    all_poi_ecef = poi_ecef
    all_poi_enu = poi_enu
    all_poi_xy = poi_xy

    # added BEGIN

    num_points = all_poi_enu.shape[0]
    labels = [chr(ord("A") + i) for i in range(num_points)]

    # 3D PLOT OF ORIGINAL 3D POINTS + BASELINE CAMERA (ONCE WE HAVE IT)

    # Compute preliminary camera estimate using all points
    k0, dist0, r0, t0 = estimate_camera_params(
        poi_enu=all_poi_enu,
        poi_xy=all_poi_xy,
        frame_size=frame_size,
        intrinsics_estimate=initial_k,
        distortion_estimate=None
    )

    cam0_enu = camera_center_enu(r0, t0)

    # Build visualization figure
    fig0 = plt.figure(figsize=(8, 6))
    ax0 = fig0.add_subplot(111, projection="3d")

    # Plot the input 3D ENU points
    ax0.scatter(
        all_poi_enu[:, 0],
        all_poi_enu[:, 1],
        all_poi_enu[:, 2],
        s=60,
        marker="x",
        color="#0080ff",
        linewidths=2,
        depthshade=False,
        label="Original 3D Points",
    )
    for i in range(num_points):
        ax0.text(
            all_poi_enu[i, 0],
            all_poi_enu[i, 1],
            all_poi_enu[i, 2] + 0.5,
            labels[i],
            color="white",
            fontsize=10,
            weight="bold",
            path_effects=[patheffects.withStroke(linewidth=2, foreground="#0080ff")],
        )

    # Plot the single estimated camera location (red X)
    ax0.scatter(
        cam0_enu[0], cam0_enu[1], cam0_enu[2],
        s=120,
        marker="x",
        color="#ff0000",
        linewidths=3,
        label="Baseline Camera",
    )

    ax0.set_xlabel("East (m)")
    ax0.set_ylabel("North (m)")
    ax0.set_zlabel("Up (m)")
    ax0.legend()
    fig0.tight_layout()

    # GIF output path
    orig_points_gif = f"{data["out"]}/original_points_spin.gif"

    make_spin_gif(
        fig=fig0,
        ax=ax0,
        gif_path=orig_points_gif,
    )

    print(f"Original points + camera GIF written to {orig_points_gif}")

    plt.close(fig0)

    # added END


    # === Perform Camera Calibration ===


    # Call the estimation function with the prepared 3D ECEF points, 2D image points,
    # camera origin GPS, frame size, and the initial intrinsic matrix guess.
    # Distortion is initially set to None, allowing estimate_camera_params to use zeros or estimate it.
    k_matrix, dist_coeffs, r_matrix, t_vector = estimate_camera_params(
        poi_enu=all_poi_enu,                 # 3D points in ENU
        poi_xy=all_poi_xy,                 # Corresponding 2D points in image
        frame_size=frame_size,             # Image dimensions (height, width)
        intrinsics_estimate=initial_k,     # Initial guess for K matrix
        distortion_estimate=None           # Initial guess for distortion (use zeros)
    )

    # added
    cam_center_enu = camera_center_enu(r_matrix, t_vector)
    cam_center_lat, cam_center_lon, cam_center_alt = pm.enu2geodetic(
        cam_center_enu[0], cam_center_enu[1], cam_center_enu[2],
        origin_gps[0], origin_gps[1], origin_gps[2]
    )
    print("\n--- Estimated Camera Location (baseline) ---")
    print(f"ENU (m): {cam_center_enu}")
    print(f"GPS (lat, lon, alt): {cam_center_lat}, {cam_center_lon}, {cam_center_alt}")

    # --- Output Estimated Parameters ---
    # The function estimate_camera_params already prints the initial guesses and reprojection error.
    # Now, print the final estimated parameters.
    print("\n--- Estimated Camera Parameters ---")
    print("Intrinsic Matrix (K):\n", k_matrix)
    print("Distortion Coefficients (D):\n", dist_coeffs.ravel()) # Flatten for easier reading
    print("Rotation Matrix (R - ENU to Camera):\n", r_matrix)
    print("Translation Vector (T - ENU to Camera):\n", t_vector.ravel())

    # --- Calculate and Output Estimated FOV ---
    # Calculate the horizontal and vertical FOV from the estimated intrinsic matrix
    hfov, vfov = calculate_fov_from_intrinsics(k_matrix, frame_size[1], frame_size[0])
    print(f"\nEstimated Horizontal FOV: {hfov:.2f} degrees")
    print(f"Estimated Vertical FOV: {vfov:.2f} degrees")

    # added BEGIN

    num_points = all_poi_enu.shape[0]


    labels = [chr(ord("A") + i) for i in range(num_points)]
    base_colors = [
        "#ff8000",  # orange
        "#cccc00",  # yellow
        "#00ff00",  # green
        "#00cccc",  # cyan
        "#0000ff",  # blue
        "#cc00cc",  # magenta
    ]
    colors = [base_colors[i % len(base_colors)] for i in range(num_points)]

    # if num_points < 7:
    #     print(f"Skipping leave-one-out; only {num_points} points, needed 7")
    # else:
    loo_centers_enu = np.zeros((num_points, 3), dtype=np.float64)
    loo_centers_gps = np.zeros((num_points, 3), dtype=np.float64)

    for i in range(num_points):
        mask = np.ones(num_points, dtype=bool)
        mask[i] = False

        poi_enu_sub = all_poi_enu[mask]
        poi_xy_sub  = all_poi_xy[mask]

        # if only 5 points → duplicate first point to make 6
        if poi_enu_sub.shape[0] < 6:
            poi_enu_sub = np.vstack([poi_enu_sub, poi_enu_sub[0]])
            poi_xy_sub  = np.vstack([poi_xy_sub,  poi_xy_sub[0]])

        k_i, dist_i, r_i, t_i = estimate_camera_params(
            poi_enu=poi_enu_sub,
            poi_xy=poi_xy_sub,
            frame_size=frame_size,
            intrinsics_estimate=initial_k,
            distortion_estimate=None
        )

        center_enu = camera_center_enu(r_i, t_i)
        loo_centers_enu[i] = center_enu

        lat_i, lon_i, alt_i = pm.enu2geodetic(
            center_enu[0], center_enu[1], center_enu[2],
            origin_gps[0], origin_gps[1], origin_gps[2]
        )
        loo_centers_gps[i] = [lat_i, lon_i, alt_i]

    loo_df = pd.DataFrame({
        "leave_out_index": np.arange(num_points),
        "enu_east_m": loo_centers_enu[:, 0],
        "enu_north_m": loo_centers_enu[:, 1],
        "enu_up_m": loo_centers_enu[:, 2],
        "lat": loo_centers_gps[:, 0],
        "lon": loo_centers_gps[:, 1],
        "alt_m": loo_centers_gps[:, 2],
    })

    loo_csv_path = f"{data["out"]}/loo_camera_centers.csv"
    loo_df.to_csv(loo_csv_path, index=False)
    print(f"\nLeave-one-out camera centers written to {loo_csv_path}")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        loo_centers_enu[:, 0],
        loo_centers_enu[:, 1],
        loo_centers_enu[:, 2],
        s=25,
        label="Leave-one-out cameras",
        c = colors,
        depthshade = False,
    )

    ax.scatter(
        cam_center_enu[0],
        cam_center_enu[1],
        cam_center_enu[2],
        s=80,
        marker="x",
        color="#ff0000",
        linewidths=3,
        label="Baseline camera",
    )

    for i in range(num_points):
        ax.text(
            loo_centers_enu[i, 0],
            loo_centers_enu[i, 1],
            loo_centers_enu[i, 2] + 0.5, # to show on top of the corresponding point
            labels[i],
            color="white",
            fontsize=10,
            weight="bold",
            path_effects=[patheffects.withStroke(linewidth=2, foreground=colors[i])],
        )

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    ax.legend()
    fig.tight_layout()

    spin_path = f"{data["out"]}/loo_camera_centers_enu_spin.gif"
    make_spin_gif(fig, ax, spin_path)
    print(f"3D ENU rotation GIF written to {spin_path}")

    loo_jpg_path = f"{data["out"]}/loo_camera_centers_enu.jpg"
    fig.savefig(loo_jpg_path, dpi=200)
    print(f"3D ENU point cloud written to {loo_jpg_path}")
    plt.close(fig)

    # === Noise perturbation analysis (part b) ===

    num_trials = 100
    noise_centers_enu = np.zeros((num_trials, 3), dtype=np.float64)
    noise_focals = np.zeros(num_trials, dtype=np.float64)

    for i in range(num_trials):
        noisy_enu = all_poi_enu + np.random.normal(0.0, 1.0, all_poi_enu.shape)
        noisy_xy = all_poi_xy + np.random.normal(0.0, 1.0, all_poi_xy.shape)

        k_n, dist_n, r_n, t_n = estimate_camera_params(
            poi_enu=noisy_enu,
            poi_xy=noisy_xy,
            frame_size=frame_size,
            intrinsics_estimate=initial_k,
            distortion_estimate=None,
            should_print = False,
        )

        noise_centers_enu[i] = camera_center_enu(r_n, t_n)
        noise_focals[i] = k_n[0, 0]

    # Robustly reject outlier focal lengths (bad calibrations)
    q1, q3 = np.percentile(noise_focals, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    valid_mask = (noise_focals >= lower) & (noise_focals <= upper)

    print(f"Dropped {np.sum(~valid_mask)} / {num_trials} hyper-noisy trials as outliers")

    fig_noise = plt.figure(figsize=(8, 6))
    ax_noise = fig_noise.add_subplot(111, projection="3d")

    ax_noise.scatter(
        noise_centers_enu[valid_mask, 0],
        noise_centers_enu[valid_mask, 1],
        noise_centers_enu[valid_mask, 2],
        s=20,
        color="#0080ff",
        depthshade=False,
        label="Noisy camera estimates (inliers)",
    )

    ax_noise.scatter(
        cam_center_enu[0],
        cam_center_enu[1],
        cam_center_enu[2],
        s=120,
        marker="x",
        color="#ff0000",
        linewidths=3,
        label="Baseline camera",
    )

    ax_noise.set_xlabel("East (m)")
    ax_noise.set_ylabel("North (m)")
    ax_noise.set_zlabel("Up (m)")
    ax_noise.legend()
    fig_noise.tight_layout()

    noise_spin_path = f"{data['out']}/noise_camera_centers_spin.gif"
    make_spin_gif(fig_noise, ax_noise, noise_spin_path)
    print(f"Noise 3D ENU rotation GIF written to {noise_spin_path}")

    plt.close(fig_noise)

    fig_hist = plt.figure(figsize=(8, 4))
    focal_eq_mm = noise_focals[valid_mask] * ff_scale
    baseline_fx_eq_mm = k_matrix[0, 0] * ff_scale

    plt.hist(focal_eq_mm, bins=20, color="#0080ff", alpha=0.8)
    plt.axvline(baseline_fx_eq_mm, color="#ff0000", linewidth=2, label="Baseline f (35mm eq.)")
    plt.xlabel("Estimated focal length f (35mm-equivalent, mm)")

    plt.ylabel("Count")
    plt.title("Distribution of focal length under noise")
    plt.legend()
    hist_path = f"{data['out']}/noise_focal_histogram.jpg"
    plt.savefig(hist_path, dpi=160, bbox_inches="tight")
    print(f"Focal length histogram written to {hist_path}")
    plt.close(fig_hist)

    # added END


    # === Reproject Points and Visualize Results ===


    # Use the estimated camera parameters (k_matrix, dist_coeffs, r_matrix, t_vector, cam_ecef_origin)
    # to project the original 3D GPS points back onto the 2D image plane.
    # This helps visualize the accuracy of the calibration.
    image_x_reprojected, image_y_reprojected, cam_distance = gps_to_camxy(
        points_gps=poi_gps,
        cam_k=k_matrix,                    # Estimated K matrix
        cam_r=r_matrix,                    # Estimated R matrix
        cam_t=t_vector,                    # Estimated T vector
        camera_gps_origin=origin_gps,      # Camera origin GPS used in calibration
        distortion=dist_coeffs             # Estimated distortion coefficients
    )

    # --- Plot the Original and Reprojected Points on the Image ---

    # Create a figure for plotting
    plt.figure(figsize=(10, 8)) # Adjust figure size as needed

    # Display the background image
    plt.imshow(default_img)

    plt.scatter(
        image_x_reprojected,
        image_y_reprojected,
        c=colors,
        s=60,
        marker="o",
        label="Reprojected POIs (Estimated Params)",
    )

    plt.scatter(
        all_poi_xy[:, 0],
        all_poi_xy[:, 1],
        c="#ff0000",
        s=80,
        marker="x",
        label="Original POIs (Input Data)",
    )

    for i in range(num_points):
        plt.text(
            all_poi_xy[i, 0] + 5,
            all_poi_xy[i, 1] - 5,
            labels[i],
            color="white",
            fontsize=11,
            weight="bold",
            path_effects=[patheffects.withStroke(linewidth=2, foreground=colors[i])],
        )

    # Add labels, title, and legend for clarity
    plt.title('Camera Calibration Results: Original vs. Reprojected Points')
    plt.xlabel('Image X Coordinate (pixels)')
    plt.ylabel('Image Y Coordinate (pixels)')
    plt.legend() # Show the legend based on the labels provided in scatter plots

    # Optionally set limits if needed, e.g., plt.xlim([0, frame_size[1]]), plt.ylim([frame_size[0], 0])
    # Set Y-axis limits to match image height, inverted because image origin is top-left
    plt.ylim([frame_size[0], 0])
    # Set X-axis limits to match image width
    plt.xlim([0, frame_size[1]])

    plt.savefig(f"{data["out"]}/image_plot.jpg", dpi = 96 * 2, bbox_inches = "tight")


def main():
    datas = [
        {
            # Define the GPS coordinates (latitude, longitude, altitude) approximately for the camera's origin point
            # This point will serve as the reference for the local East-North-Up (ENU) coordinate system
            "origin": {
                "latitude": 43.07063697146,
                "longitude": -89.40685704184578,
                "altitude": 263 + 204*0.3048,
            },
            # Construct the full path to the CSV file containing point correspondences
            # Using the raw content URL from GitHub for direct access
            "path_to_csv": 'https://raw.githubusercontent.com/shrnik/contrails/main/uwisc/east/matches.csv',
            "url": 'https://raw.githubusercontent.com/shrnik/contrails/main/uwisc/east/DEFAULT.jpg',
            # Define an initial guess for the camera's vertical field of view (FOV) in degrees.
            # This is often known approximately from camera specifications or can be estimated.
            "vfov_deg_estimate": 63.59, # Example estimated vertical FOV
            "out": "wisconsin",
        },
        {
            "origin": {
                "latitude": 42.352397,
                "longitude": -71.048881,
                "altitude": 14 + 14,
            },
            "path_to_csv": "./boston_points.csv",
            "vfov_deg_estimate": 85,
            "url": 'http://sleeper.dyndns.org/record/current.jpg',
            "out": "boston",
        },
    ]

    for data in datas:
        process_data(data)


if __name__ == "__main__":
    main()
