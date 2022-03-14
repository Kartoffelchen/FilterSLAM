
from lego_robot import LegoLogfile
from pf_library import get_cylinders_from_scan, assign_cylinders
from math import sin, cos, pi, atan2, sqrt
import random
from scipy.stats import norm as normal_dist


class ParticleFilter:
    def __init__(self, initial_particles,
                 robot_width, scanner_displacement,
                 control_motion_factor, control_turn_factor,
                 measurement_distance_stddev, measurement_angle_stddev):
        # The particles.
        self.particles = initial_particles

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

        return (g1, g2, g3)

    def predict(self, control):
        l, r = control
        w=self.robot_width
        alpha1=self.control_motion_factor
        alpha2=self.control_turn_factor
        sigma_l=sqrt((alpha1*l)**2+(alpha2*(l-r))**2)
        sigma_r=sqrt((alpha1*r)**2+(alpha2*(l-r))**2)
        new_particles_list=[]
        control_prime=[]
        for i in self.particles:
            l_prime=random.gauss(l,sigma_l)
            r_prime=random.gauss(r,sigma_r)
            control_prime=[l_prime,r_prime]
            new_particles_list.append(self.g(i,control_prime,w))
        self.particles=new_particles_list
        
    @staticmethod
    def h(state, landmark, scanner_displacement):
        dx = landmark[0] - (state[0] + scanner_displacement * cos(state[2]))
        dy = landmark[1] - (state[1] + scanner_displacement * sin(state[2]))
        r = sqrt(dx * dx + dy * dy)
        alpha = (atan2(dy, dx) - state[2] + pi) % (2*pi) - pi
        return (r, alpha)

    def probability_of_measurement(self, measurement, predicted_measurement):
        return  normal_dist.pdf(predicted_measurement[0]-measurement[0],0,self.measurement_distance_stddev)*normal_dist.pdf(predicted_measurement[1]-measurement[1],0,self.measurement_angle_stddev)

    def compute_weights(self, cylinders, landmarks):
        weights = []
        
        for p in self.particles:
            assignment = assign_cylinders(cylinders, p, self.scanner_displacement, landmarks)
            probility=1
            for i in assignment:
                predicted1,predicted2=self.h(p,i[1],self.scanner_displacement)
                probility=probility*self.probability_of_measurement((predicted1,predicted2),i[0])
            weights.append(probility) 
        return weights

    def resample(self, weights):
        new_particles=[]
        offset=0.0
        max_weight=max(weights)
        index=random.randint(0,len(weights)-1)
        for i in range(len(weights)):
            offset+=random.uniform(0,2*max_weight)
            while(offset>weights[index]):
                offset-=weights[index]
                index = (index+1)%len(weights)
            new_particles.append(self.particles[index])
        return new_particles

    def correct(self, cylinders, landmarks):
        weights = self.compute_weights(cylinders, landmarks)
        self.particles = self.resample(weights)

    def print_particles(self, file_desc):
        if not self.particles:
            return
        file_desc.writelines( "PA " )
        for p in self.particles:
            file_desc.writelines( "%.0f %.0f %.3f" % p)
        file_desc.writelines( "\n" )

    def get_mean(self):
        sum_x=0.0
        sum_y=0.0
        v_x=0.0
        v_y=0.0
        for i in self.particles:
            sum_x+=i[0]
            sum_y+=i[1]
            v_x+=cos(i[2])
            v_y+=sin(i[2])
        mean_x=sum_x/len(self.particles)
        mean_y=sum_y/len(self.particles)
        mean_heading=atan2(v_y,v_x)
        return (mean_x, mean_y, mean_heading)  # Replace this.


if __name__ == '__main__':
    # Robot constants.
    scanner_displacement = 30.0
    ticks_to_mm = 0.349
    robot_width = 155.0

    # Cylinder extraction and matching constants.
    minimum_valid_distance = 20.0
    depth_jump = 100.0
    cylinder_offset = 90.0

    # Filter constants.
    control_motion_factor = 0.35 
    control_turn_factor = 0.6 
    measurement_distance_stddev = 200.0 
    measurement_angle_stddev = 15.0 / 180.0 * pi 

    number_of_particles = 50
    measured_state = (1850.0, 1897.0, 213.0 / 180.0 * pi)
    standard_deviations = (100.0, 100.0, 10.0 / 180.0 * pi)
    initial_particles = []
    for i in range(number_of_particles):
        initial_particles.append(tuple([
            random.gauss(measured_state[j], standard_deviations[j])
            for j in range(3)]))

    pf = ParticleFilter(initial_particles,
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

    f = open("particle_filter_mean.txt", "w")
    for i in range(len(logfile.motor_ticks)):
        control = map(lambda x: x * ticks_to_mm, logfile.motor_ticks[i])
        pf.predict(control)

        cylinders = get_cylinders_from_scan(logfile.scan_data[i], depth_jump,
            minimum_valid_distance, cylinder_offset)
        pf.correct(cylinders, reference_cylinders)

        pf.print_particles(f)
        
        mean = pf.get_mean()
        f.writelines( "F %.0f %.0f %.3f" %\
              (mean[0] + scanner_displacement * cos(mean[2]),
               mean[1] + scanner_displacement * sin(mean[2]),
               mean[2]))
        f.writelines( "\n" )

    f.close()
