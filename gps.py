import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P, x):
        """
        Initializes the Kalman Filter.
        
        Parameters:
        - A: State transition matrix
        - B: Control input matrix
        - H: Observation matrix
        - Q: Process noise covariance
        - R: Measurement noise covariance
        - P: Initial estimation error covariance
        - x: Initial state estimate
        """
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x
        self.K_history = []  # To store Kalman gain history

    def predict(self, u):
        """
        Predicts the next state and estimation error covariance.
        
        Parameters:
        - u: Control input
        """
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        """
        Updates the state estimate and estimation error covariance using the measurement z.
        
        Parameters:
        - z: Measurement vector
        """
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.A.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        self.K_history.append(K)  # Store Kalman gain

def simulate_data(num_steps, dt, initial_position, initial_velocity, acceleration, gps_noise_std):
    """
    Simulates true state and noisy GPS measurements.
    
    Parameters:
    - num_steps: Number of time steps
    - dt: Time step duration
    - initial_position: Initial position of the vehicle
    - initial_velocity: Initial velocity of the vehicle
    - acceleration: Constant acceleration of the vehicle
    - gps_noise_std: Standard deviation of GPS measurement noise
    
    Returns:
    - true_positions: Array of true positions
    - true_velocities: Array of true velocities
    - measurements: Array of GPS measurements
    """
    true_positions = []
    true_velocities = []
    measurements = []

    position = initial_position
    velocity = initial_velocity

    for _ in range(num_steps):
        # Update true state
        velocity += acceleration * dt
        position += velocity * dt

        true_positions.append(position)
        true_velocities.append(velocity)

        # Simulate GPS measurement with noise
        gps_measurement = position + np.random.normal(0, gps_noise_std)
        measurements.append(gps_measurement)

    return np.array(true_positions), np.array(true_velocities), np.array(measurements)

def main():
    # Simulation parameters
    num_steps = 50
    dt = 1.0  # time step (seconds)
    initial_position = 0.0  # meters
    initial_velocity = 20.0  # meters/second
    acceleration = 1.0  # meters/second^2
    gps_noise_std = 10.0  # meters

    # Simulate true state and measurements
    true_positions, true_velocities, measurements = simulate_data(
        num_steps, dt, initial_position, initial_velocity, acceleration, gps_noise_std
    )

    # Define Kalman Filter matrices
    A = np.array([[1, dt],
                  [0, 1]])  # State transition matrix

    B = np.array([[0.5 * dt**2],
                  [dt]])  # Control input matrix

    H = np.array([[1, 0]])  # Observation matrix

    Q = np.array([[1, 0],
                  [0, 1]])  # Process noise covariance

    R = np.array([[gps_noise_std**2]])  # Measurement noise covariance

    P = np.array([[1000, 0],
                  [0, 1000]])  # Initial estimation error covariance

    x_initial = np.array([[0],
                          [0]])  # Initial state estimate

    # Control input (acceleration)
    u = np.array([[acceleration]])

    # Initialize Kalman Filter
    kf = KalmanFilter(A, B, H, Q, R, P, x_initial)

    # Lists to store estimates
    estimated_positions = []
    estimated_velocities = []

    for i in range(num_steps):
        # Predict
        kf.predict(u)

        # Update with measurement
        z = np.array([[measurements[i]]])
        kf.update(z)

        # Store estimates
        estimated_positions.append(kf.x[0, 0])
        estimated_velocities.append(kf.x[1, 0])

    # Plotting the results
    time_steps = np.arange(num_steps) * dt

    plt.figure(figsize=(12, 6))

    # Position plot
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, true_positions, label='True Position', color='g')
    plt.scatter(time_steps, measurements, label='GPS Measurements', color='r', marker='x')
    plt.plot(time_steps, estimated_positions, label='Kalman Filter Estimate', color='b')
    plt.title('Kalman Filter Position Estimation')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.legend()
    plt.grid(True)

    # Velocity plot
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, true_velocities, label='True Velocity', color='g')
    plt.plot(time_steps, estimated_velocities, label='Kalman Filter Estimate', color='b')
    plt.title('Kalman Filter Velocity Estimation')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.legend()
    plt.grid(True)

    # Kalman gain plot
    plt.figure(figsize=(12, 6))
    kalman_gains = np.array(kf.K_history).squeeze()
    plt.plot(time_steps, kalman_gains[:, 0], label='Kalman Gain')
    plt.title('Kalman Gain')
    plt.xlabel('Time [s]')
    plt.ylabel('Kalman Gain')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()