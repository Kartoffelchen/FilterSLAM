
from lego_robot import LegoLogfile
from pf_slam_library import get_cylinders_from_scan, write_cylinders,\
     write_error_ellipses, get_mean, get_error_ellipse_and_heading_variance,\
     print_particles
from math import sin, cos, pi, atan2, sqrt, exp
import copy
import random
import numpy as np


class Particle:
    def __init__(self, pose):
        self.pose = pose
        self.landmark_positions = []
        self.landmark_covariances = []

    def number_of_landmarks(self):
        return len(self.landmark_positions)

    @staticmethod
    def g(state, control, w):
        x, y, theta = state
        l, r = control
        if r != l:
            alpha = (r - l) / w
            rad = l/alpha
            g1 = x + (rad + w/2.)*(sin(theta+alpha) - sin(theta))
            g2 = y + (rad + w/2.)*(-cos(theta+alpha) + cos(theta))
            g3 = (theta + alpha + pi) % (2*pi) - pi
        else:
            g1 = x + l * cos(theta)
            g2 = y + l * sin(theta)
            g3 = theta
        return np.array([g1, g2, g3])

    def move(self, left, right, w):
        self.pose = self.g(self.pose, (left, right), w)

    @staticmethod
    def h(state, landmark, scanner_displacement):
        dx = landmark[0] - (state[0] + scanner_displacement * cos(state[2]))
        dy = landmark[1] - (state[1] + scanner_displacement * sin(state[2]))
        r = sqrt(dx * dx + dy * dy)
        alpha = (atan2(dy, dx) - state[2] + pi) % (2*pi) - pi
        return np.array([r, alpha])

    @staticmethod
    def dh_dlandmark(state, landmark, scanner_displacement):
        theta = state[2]
        cost, sint = cos(theta), sin(theta)
        dx = landmark[0] - (state[0] + scanner_displacement * cost)
        dy = landmark[1] - (state[1] + scanner_displacement * sint)
        q = dx * dx + dy * dy
        sqrtq = sqrt(q)
        dr_dmx = dx / sqrtq
        dr_dmy = dy / sqrtq
        dalpha_dmx = -dy / q
        dalpha_dmy =  dx / q

        return np.array([[dr_dmx, dr_dmy],
                         [dalpha_dmx, dalpha_dmy]])

    def h_expected_measurement_for_landmark(self, landmark_number,
                                            scanner_displacement):
        r, alpha = self.h(self.pose, self.landmark_positions[landmark_number], scanner_displacement)
        return np.array([[r,alpha],[self.pose]])  # Replace this.

    def H_Ql_jacobian_and_measurement_covariance_for_landmark(
        self, landmark_number, Qt_measurement_covariance, scanner_displacement):
        H = self.dh_dlandmark(self.pose, self.landmark_positions[landmark_number], scanner_displacement)
        Ql = np.dot(H,np.dot(self.landmark_covariances[landmark_number],H.T))+Qt_measurement_covariance
        return (H, Ql)

    def wl_likelihood_of_correspondence(self, measurement,
                                        landmark_number,
                                        Qt_measurement_covariance,
                                        scanner_displacement):
        h =  self.h_expected_measurement_for_landmark(landmark_number,scanner_displacement)
        delta_z = measurement - h[0]
        H,Ql = self.H_Ql_jacobian_and_measurement_covariance_for_landmark(landmark_number, Qt_measurement_covariance, scanner_displacement)
        wl = 1/(2*pi*np.linalg.det(Ql)**0.5)*exp(-0.5*np.dot(delta_z.T,np.dot(np.linalg.inv(Ql),delta_z)))
        return wl

    def compute_correspondence_likelihoods(self, measurement,
                                           number_of_landmarks,
                                           Qt_measurement_covariance,
                                           scanner_displacement):
        likelihoods = []
        for i in range(number_of_landmarks):
            likelihoods.append(
                self.wl_likelihood_of_correspondence(
                    measurement, i, Qt_measurement_covariance,
                    scanner_displacement))
        return likelihoods

    def initialize_new_landmark(self, measurement_in_scanner_system,
                                Qt_measurement_covariance,
                                scanner_displacement):
        scanner_pose = (self.pose[0] + cos(self.pose[2]) * scanner_displacement,
                        self.pose[1] + sin(self.pose[2]) * scanner_displacement,
                        self.pose[2])
        landmark_x,landmark_y = LegoLogfile.scanner_to_world(scanner_pose,measurement_in_scanner_system)
        H = self.dh_dlandmark(self.pose, [landmark_x,landmark_y], scanner_displacement)
        landmark_covariance = np.dot(np.linalg.inv(H),np.dot(Qt_measurement_covariance,np.linalg.inv(H).T))
        self.landmark_positions.append(np.array([0.0, 0.0]))
        self.landmark_covariances.append(np.eye(2))

    def update_landmark(self, landmark_number, measurement,
                        Qt_measurement_covariance, scanner_displacement):
        h =  self.h_expected_measurement_for_landmark(landmark_number,scanner_displacement)
        delta_z = measurement - h[0]
        H,Ql = self.H_Ql_jacobian_and_measurement_covariance_for_landmark(landmark_number, Qt_measurement_covariance, scanner_displacement)        
        K = np.dot(np.dot(self.landmark_covariances[landmark_number],H.T),np.linalg.inv(Ql))
        self.landmark_positions[landmark_number] += np.dot(K,delta_z) 
        self.landmark_covariances[landmark_number] = np.dot((np.eye(np.dot(K,H)[0].size)-np.dot(K,H)),self.landmark_covariances[landmark_number])

    def update_particle(self, measurement, measurement_in_scanner_system,
                        number_of_landmarks,
                        minimum_correspondence_likelihood,
                        Qt_measurement_covariance, scanner_displacement):
        likelihoods = self.compute_correspondence_likelihoods(measurement,number_of_landmarks,Qt_measurement_covariance,scanner_displacement)

        if not likelihoods or max(likelihoods) < minimum_correspondence_likelihood:
            self.initialize_new_landmark(measurement_in_scanner_system,Qt_measurement_covariance,scanner_displacement)
            return minimum_correspondence_likelihood

        else:
            w, index = max(likelihoods), likelihoods.index(max(likelihoods)) 
            self.update_landmark(index, measurement,Qt_measurement_covariance, scanner_displacement)
            return w

class FastSLAM:
    def __init__(self, initial_particles,
                 robot_width, scanner_displacement,
                 control_motion_factor, control_turn_factor,
                 measurement_distance_stddev, measurement_angle_stddev,
                 minimum_correspondence_likelihood):
        self.particles = initial_particles

        self.robot_width = robot_width
        self.scanner_displacement = scanner_displacement
        self.control_motion_factor = control_motion_factor
        self.control_turn_factor = control_turn_factor
        self.measurement_distance_stddev = measurement_distance_stddev
        self.measurement_angle_stddev = measurement_angle_stddev
        self.minimum_correspondence_likelihood = \
            minimum_correspondence_likelihood

    def predict(self, control):
        left, right = control
        left_std  = sqrt((self.control_motion_factor * left)**2 + (self.control_turn_factor * (left-right))**2)
        right_std = sqrt((self.control_motion_factor * right)**2 +  (self.control_turn_factor * (left-right))**2)
        for p in self.particles:
            l = random.gauss(left, left_std)
            r = random.gauss(right, right_std)
            p.move(l, r, self.robot_width)

    def update_and_compute_weights(self, cylinders):
        Qt_measurement_covariance = \
            np.diag([self.measurement_distance_stddev**2,
                     self.measurement_angle_stddev**2])
        weights = []
        for p in self.particles:
            number_of_landmarks = p.number_of_landmarks()
            weight = 1.0
            for measurement, measurement_in_scanner_system in cylinders:
                weight *= p.update_particle(
                    measurement, measurement_in_scanner_system,
                    number_of_landmarks,
                    self.minimum_correspondence_likelihood,
                    Qt_measurement_covariance, scanner_displacement)
            weights.append(weight)
        return weights

    def resample(self, weights):
        new_particles = []
        max_weight = max(weights)
        index = random.randint(0, len(self.particles) - 1)
        offset = 0.0
        for i in range(len(self.particles)):
            offset += random.uniform(0, 2.0 * max_weight)
            while offset > weights[index]:
                offset -= weights[index]
                index = (index + 1) % len(weights)
            new_particles.append(copy.deepcopy(self.particles[index]))
        return new_particles

    def correct(self, cylinders):
        weights = self.update_and_compute_weights(cylinders)
        self.particles = self.resample(weights)


if __name__ == '__main__':
    # Robot constants.
    scanner_displacement = 30.0
    ticks_to_mm = 0.349
    robot_width = 155.0

    minimum_valid_distance = 20.0
    depth_jump = 100.0
    cylinder_offset = 90.0

    control_motion_factor = 0.35 
    control_turn_factor = 0.6 
    measurement_distance_stddev = 200.0 
    measurement_angle_stddev = 15.0 / 180.0 * pi 
    minimum_correspondence_likelihood = 0.001 

    number_of_particles = 25
    start_state = np.array([500.0, 0.0, 45.0 / 180.0 * pi])
    initial_particles = [copy.copy(Particle(start_state))
                         for _ in range(number_of_particles)]

    fs = FastSLAM(initial_particles,
                  robot_width, scanner_displacement,
                  control_motion_factor, control_turn_factor,
                  measurement_distance_stddev,
                  measurement_angle_stddev,
                  minimum_correspondence_likelihood)

    logfile = LegoLogfile()
    logfile.read("robot4_motors.txt")
    logfile.read("robot4_scan.txt")

    f = open("fast_slam_correction.txt", "w")
    for i in range(len(logfile.motor_ticks)):
        control = map(lambda x: x * ticks_to_mm, logfile.motor_ticks[i])
        fs.predict(control)

        cylinders = get_cylinders_from_scan(logfile.scan_data[i], depth_jump,
            minimum_valid_distance, cylinder_offset)
        fs.correct(cylinders)

        print_particles(fs.particles, f)

        mean = get_mean(fs.particles)        
        f.writelines( "F %.0f %.0f %.3f" %\
              (mean[0] + scanner_displacement * cos(mean[2]),
               mean[1] + scanner_displacement * sin(mean[2]),
               mean[2]))
        f.writelines("\n")
        
        errors = get_error_ellipse_and_heading_variance(fs.particles, mean)
        f.writelines( "E %.3f %.0f %.0f %.3f" % errors )
        f.writelines("\n")
        

        output_particle = min([
            (np.linalg.norm(mean[0:2] - fs.particles[i].pose[0:2]),i)
            for i in range(len(fs.particles)) ])[1]
        
        write_cylinders(f, "W C ",
                        fs.particles[output_particle].landmark_positions)
                        
        
        write_error_ellipses(f, "W E ",
                             fs.particles[output_particle].landmark_covariances)

        
    f.close()
