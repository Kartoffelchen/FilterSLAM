
from math import sin, cos, pi, atan2, sqrt
from numpy import *
from kf_library import get_observations, write_cylinders
from lego_robot import LegoLogfile


class ExtendedKalmanFilter:
    def __init__(self, state, covariance,
                 robot_width, scanner_displacement,
                 control_motion_factor, control_turn_factor,
                 measurement_distance_stddev, measurement_angle_stddev):
        # The state. This is the important data of the Kalman filter.
        self.state = state
        self.covariance =  covariance

        # Some constants.
        self.robot_width = robot_width
        self.scanner_displacement = scanner_displacement
        self.control_motion_factor = control_motion_factor
        self.control_turn_factor = control_turn_factor
        self.measurement_distance_stddev = measurement_distance_stddev
        self.measurement_angle_stddev = measurement_angle_stddev

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

        return array([g1, g2, g3])

    @staticmethod
    def dg_dstate(state, control, w):
        theta = state[2]
        l, r = control
        if r != l:
            alpha = (r - l) / w
            rad = l/alpha
            dg1_dx=1
            dg1_dy=0
            dg1_dtheta=(rad+w/2)*(cos(theta+alpha)-cos(theta))
            dg2_dx=0
            dg2_dy=1
            dg2_dtheta=(rad+w/2)*(sin(theta+alpha)-sin(theta))
            dg3_dx=0
            dg3_dy=0
            dg3_dtheta=1
            
            m = array([[dg1_dx, dg1_dy, dg1_dtheta], [dg2_dx, dg2_dy, dg2_dtheta], [dg3_dx, dg3_dy, dg3_dtheta]]) 

        else:
            m = array([[1, 0, -l*sin(theta)], [0, 1, l*cos(theta)], [0, 0, 1]]) 
        return m

    @staticmethod
    def dg_dcontrol(state, control, w):
        theta = state[2]
        l, r = tuple(control)
        if r != l:
            alpha = (r - l) / w
            theta_p=theta+alpha
            dg1_dl=w*r*(sin(theta_p)-sin(theta))/((r-l)**2) - (r+l)*cos(theta_p)/(2*(r-l))
            dg2_dl=w*r*(-cos(theta_p)+cos(theta))/((r-l)**2) - (r+l)*sin(theta_p)/(2*(r-l))
            dg3_dl=-1/w
            dg1_dr=-w*l*(sin(theta_p)-sin(theta))/((r-l)**2)+(r+l)*cos(theta_p)/(2*(r-l))
            dg2_dr=-w*l*(-cos(theta_p)+cos(theta))/((r-l)**2)+(r+l)*sin(theta_p)/(2*(r-l))
            dg3_dr=1/w
                       
        else:

            dg1_dl=0.5*(cos(theta)+l/w*sin(theta))
            dg2_dl=0.5*(sin(theta)-l/w*cos(theta))
            dg3_dl=-1/w
            dg1_dr=0.5*(-l/w*sin(theta)+cos(theta))
            dg2_dr=0.5*(l/w*cos(theta)+sin(theta))
            dg3_dr=1/w           

        m = array([[dg1_dl,dg1_dr], [dg2_dl, dg2_dr], [dg3_dl, dg3_dr]])            
        return m

    @staticmethod
    def get_error_ellipse(covariance):
        eigenvals, eigenvects = linalg.eig(covariance[0:2,0:2])
        angle = atan2(eigenvects[1,0], eigenvects[0,0])
        return (angle, sqrt(eigenvals[0]), sqrt(eigenvals[1]))        

    def predict(self, control):
        left, right = control
        variance_l2=(self.control_motion_factor*left)**2+(self.control_turn_factor*(left-right))**2
        variance_r2=(self.control_motion_factor*right)**2+(self.control_turn_factor*(left-right))**2
        V=self.dg_dcontrol(self.state,control,self.robot_width)
        R=dot(dot(V,diag([variance_l2,variance_r2])),V.T)
        G=self.dg_dstate(self.state,control,self.robot_width)
        self.covariance=dot(dot(G,self.covariance),G.T)+R
        self.state=self.g(self.state,control,self.robot_width)

    @staticmethod
    def h(state, landmark, scanner_displacement):
        dx = landmark[0] - (state[0] + scanner_displacement * cos(state[2]))
        dy = landmark[1] - (state[1] + scanner_displacement * sin(state[2]))
        r = sqrt(dx * dx + dy * dy)
        alpha = (atan2(dy, dx) - state[2] + pi) % (2*pi) - pi
        return array([r, alpha])

    @staticmethod
    def dh_dstate(state, landmark, scanner_displacement):
        x=state[0]
        y=state[1]
        theta=state[2]
        x_m=landmark[0]
        y_m=landmark[1]
        xl=x+scanner_displacement*cos(theta)
        yl=y+scanner_displacement*sin(theta)
        delta_x=x_m-xl
        delta_y=y_m-yl
        q=delta_x**2+delta_y**2
        dr_dx=-delta_x/sqrt(q)
        dr_dy=-delta_y/sqrt(q)
        dr_dtheta=scanner_displacement/sqrt(q)*(delta_x*sin(theta)-delta_y*cos(theta))
        dalpha_dx=delta_y/q
        dalpha_dy=-delta_x/q
        dalpha_dtheta=-scanner_displacement/q*(delta_x*cos(theta)+delta_y*sin(theta))-1
        return array([[dr_dx, dr_dy, dr_dtheta], [dalpha_dx, dalpha_dy, dalpha_dtheta]]) 

    def correct(self, measurement, landmark):
        H=self.dh_dstate(self.state,landmark,self.scanner_displacement)
        Q=diag([self.measurement_distance_stddev**2,self.measurement_angle_stddev**2])
        Inner=linalg.inv(dot(H,dot(self.covariance,H.T))+Q)
        K=dot(self.covariance,dot(H.T,Inner))
        innovation = array(measurement) -self.h(self.state, landmark, self.scanner_displacement)
        innovation[1] = (innovation[1] + pi) % (2*pi) - pi
        self.state = self.state + dot(K,innovation)
        self.covariance = dot((eye(3) - dot(K,H)),self.covariance)

if __name__ == '__main__':
    # Robot constants.
    scanner_displacement = 30.0
    ticks_to_mm = 0.349
    robot_width = 155.0

    # Cylinder extraction and matching constants.
    minimum_valid_distance = 20.0
    depth_jump = 100.0
    cylinder_offset = 90.0
    max_cylinder_distance = 300.0

    # Filter constants.
    control_motion_factor = 0.35 
    control_turn_factor = 0.6 
    measurement_distance_stddev = 200.0  
    measurement_angle_stddev = 15.0 / 180.0 * pi  

    # Measured start position.
    initial_state = array([1850.0, 1897.0, 213.0 / 180.0 * pi])
    # Covariance at start position.
    initial_covariance = diag([100.0**2, 100.0**2, (10.0 / 180.0 * pi) ** 2])
    # Setup filter.
    kf = ExtendedKalmanFilter(initial_state, initial_covariance,
                              robot_width, scanner_displacement,
                              control_motion_factor, control_turn_factor,
                              measurement_distance_stddev,
                              measurement_angle_stddev)

    # Read data.
    logfile = LegoLogfile()
    logfile.read("robot4_motors.txt")
    logfile.read("robot4_scan.txt")
    logfile.read("robot_arena_landmarks.txt")
    reference_cylinders = [l[1:3] for l in logfile.landmarks]

    states = []
    covariances = []
    matched_ref_cylinders = []
    for i in range(len(logfile.motor_ticks)):
        control = array(logfile.motor_ticks[i]) * ticks_to_mm
        kf.predict(control)

        # Correction.
        observations = get_observations( 
            logfile.scan_data[i],
            depth_jump, minimum_valid_distance, cylinder_offset,
            kf.state, scanner_displacement,
            reference_cylinders, max_cylinder_distance)
        for j in range(len(observations)):
            kf.correct(*observations[j])

        states.append(kf.state)
        covariances.append(kf.covariance)
        matched_ref_cylinders.append([m[1] for m in observations])

    f = open("kalman_prediction_and_correction.txt", "w")
    for i in range(len(states)):
        f.writelines( "F %f %f %f" % \
            tuple(states[i] + [scanner_displacement * cos(states[i][2]),
                               scanner_displacement * sin(states[i][2]),
                               0.0]))
        f.writelines( "\n")
        
        e = ExtendedKalmanFilter.get_error_ellipse(covariances[i])
        f.writelines( "E %f %f %f %f" % (e + (sqrt(covariances[i][2,2]),)))
        f.writelines( "\n")
        
        write_cylinders(f, "W C", matched_ref_cylinders[i])        

    f.close()
