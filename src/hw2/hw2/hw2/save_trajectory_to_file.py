import datetime
import os

def save_trajectory_to_file(trajectory, logger=None):
    """
    Save trajectory to file
    Args:
        trajectory: list of 6×1 joint vectors, each element is a list of 6 joint angles at time t_i
        logger: optional logger for output messages
    """
    try:
        # Get the src directory
        # When running via ros2 run, __file__ points to install directory:
        # /ros2_ws/install/hw2/lib/python3.10/site-packages/hw2/save_trajectory_to_file.py
        # We need to find /ros2_ws/src/
        current_file = os.path.abspath(__file__)
        path = current_file
        src_dir = None
        
        # Go up the directory tree to find workspace root (contains 'src' directory)
        # Look for a directory that contains both 'src' and 'install' subdirectories
        for _ in range(15):  # Limit search depth
            parent = os.path.dirname(path)
            src_path = os.path.join(parent, 'src')
            install_path = os.path.join(parent, 'install')
            # Check if this looks like a ROS2 workspace (has both src and install)
            if os.path.exists(src_path) and os.path.isdir(src_path) and \
               os.path.exists(install_path) and os.path.isdir(install_path):
                src_dir = src_path
                break
            if parent == path:  # Reached filesystem root
                break
            path = parent
        
        # Fallback: if workspace not found, try current working directory
        if src_dir is None:
            cwd = os.getcwd()
            # Check if current directory or parent has src
            if os.path.exists(os.path.join(cwd, 'src')) and os.path.isdir(os.path.join(cwd, 'src')):
                src_dir = os.path.join(cwd, 'src')
            else:
                # Last resort: use current directory
                src_dir = cwd
                if logger:
                    logger.warn(f'Could not find workspace src directory, using: {src_dir}')
        
        filename = os.path.join(src_dir, f'trajectory_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        # Save trajectory
    with open(filename, 'w') as f:
            for t_i, joint_vector in enumerate(trajectory):
                # Each line: 6 joint angles separated by commas
                # Format: angle1,angle2,angle3,angle4,angle5,angle6
                f.write(f'{joint_vector[0]:.6f},{joint_vector[1]:.6f},{joint_vector[2]:.6f},{joint_vector[3]:.6f},{joint_vector[4]:.6f},{joint_vector[5]:.6f}\n')
        
        # Log success message
        if logger:
            logger.info(f'Trajectory saved to file: {filename} ({len(trajectory)} points)')
        else:
            print(f'Trajectory saved to file: {filename} ({len(trajectory)} points)')
        
        return filename
    except Exception as e:
        error_msg = f'Error saving trajectory to file: {str(e)}'
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        raise

if __name__ == '__main__':
    trajectory = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # trajectory of joint 1
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # trajectory of joint 2
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # trajectory of joint 3
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # trajectory of joint 4
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # trajectory of joint 5
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # trajectory of joint 6
    ]
    save_trajectory_to_file(trajectory)
