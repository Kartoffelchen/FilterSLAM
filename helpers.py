from math import sin, cos, pi

s_record_has_count = True

class LegoLogfile(object):
    def __init__(self):
        self.reference_positions = []
        self.scan_data = []
        self.pole_indices = []
        self.motor_ticks = []
        self.filtered_positions = []
        self.filtered_stddev = []
        self.landmarks = []
        self.detected_cylinders = []
        self.world_cylinders = []
        self.last_ticks = None

    def read(self, filename):        
        first_reference_positions = True
        first_scan_data = True
        first_pole_indices = True
        first_motor_ticks = True
        first_filtered_positions = True
        first_filtered_stddev = True
        first_landmarks = True
        first_detected_cylinders = True
        first_world_cylinders = True
        f = open(filename)
        
        for l in f:
            sp = l.split()
            
            if sp[0] == 'P':
                if first_reference_positions:
                    self.reference_positions = []
                    first_reference_positions = False 
                self.reference_positions.append( (int(sp[2]), int(sp[3])) )

            
            elif sp[0] == 'S':
                if first_scan_data:
                    self.scan_data = []
                    first_scan_data = False
                if s_record_has_count:
                    self.scan_data.append(tuple(map(int, sp[3:])))
                else:
                    self.scan_data.append(tuple(map(int, sp[2:])))

            
            elif sp[0] == 'I':
                if first_pole_indices:
                    self.pole_indices = []
                    first_pole_indices = False
                self.pole_indices.append(tuple(map(int, sp[2:])))

            
            elif sp[0] == 'M':
                ticks = (int(sp[2]), int(sp[6]))
                if first_motor_ticks:
                    self.motor_ticks = []
                    first_motor_ticks = False
                    self.last_ticks = ticks
                self.motor_ticks.append(
                    tuple([ticks[i]-self.last_ticks[i] for i in range(2)]))
                self.last_ticks = ticks

            
            elif sp[0] == 'F':
                if first_filtered_positions:
                    self.filtered_positions = []
                    first_filtered_positions = False
                self.filtered_positions.append( tuple( map(float, sp[1:])) )

            
            elif sp[0] == 'E':
                if first_filtered_stddev:
                    self.filtered_stddev = []
                    first_filtered_stddev = False
                self.filtered_stddev.append( tuple( map(float, sp[1:])) )
                
            
            elif sp[0] == 'L':
                if first_landmarks:
                    self.landmarks = []
                    first_landmarks = False
                if sp[1] == 'C':
                    self.landmarks.append( tuple(['C'] + map(float, sp[2:])) )
                    
            
            elif sp[0] == 'D':
                if sp[1] == 'C':
                    if first_detected_cylinders:
                        self.detected_cylinders = []
                        first_detected_cylinders = False
                    cyl = map(float, sp[2:])
                    self.detected_cylinders.append([(cyl[2*i], cyl[2*i+1]) for i in range(len(cyl)/2)])

            
            elif sp[0] == 'W':
                if sp[1] == 'C':
                    if first_world_cylinders:
                        self.world_cylinders = []
                        first_world_cylinders = False
                    cyl = map(float, sp[2:])
                    self.world_cylinders.append([(cyl[2*i], cyl[2*i+1]) for i in range(len(cyl)/2)])

        f.close()

    def size(self):        
        return max(len(self.reference_positions), len(self.scan_data),
                   len(self.pole_indices), len(self.motor_ticks),
                   len(self.filtered_positions), len(self.filtered_stddev),
                   len(self.detected_cylinders), len(self.world_cylinders))

    @staticmethod
    def beam_index_to_angle(i, mounting_angle = -0.06981317007977318):
        return (i - 330.0) * 0.006135923151543 + mounting_angle

    @staticmethod
    def scanner_to_world(pose, point):           
        dx = cos(pose[2])
        dy = sin(pose[2])
        x, y = point
        return (x * dx - y * dy + pose[0], x * dy + y * dx + pose[1])        

    def info(self, i):
        s = ""
        if i < len(self.reference_positions):
            s += " | ref-pos: %4d %4d" % self.reference_positions[i]

        if i < len(self.scan_data):
            s += " | scan-points: %d" % len(self.scan_data[i])

        if i < len(self.pole_indices):
            indices = self.pole_indices[i]
            if indices:
                s += " | pole-indices:"
                for idx in indices:
                    s += " %d" % idx
            else:
                s += " | (no pole indices)"
                    
        if i < len(self.motor_ticks):
            s += " | motor: %d %d" % self.motor_ticks[i]

        if i < len(self.filtered_positions):
            f = self.filtered_positions[i]
            s += " | filtered-pos:"
            for j in (0,1):
                s += " %.1f" % f[j]
            if len(f) > 2:
                s += " %.1f" % (f[2] / pi * 180.)

        if i < len(self.filtered_stddev):
            stddev = self.filtered_stddev[i]
            s += " | stddev:"
            for j in (1,2):
                s += " %.1f" % stddev[j]
            if len(stddev) > 3:
                s += " %.1f" % (stddev[3] / pi * 180.)

        return s
        
        
        
        

def get_cylinders_from_scan(scan, jump, min_dist, cylinder_offset): 
    der = compute_derivative(scan, min_dist)
    cylinders = find_cylinders(scan, der, jump, min_dist)
    result = []
    for c in cylinders:
        bearing = LegoLogfile.beam_index_to_angle(c[0])
        distance = c[1] + cylinder_offset
        x, y = distance*cos(bearing), distance*sin(bearing)
        result.append( (distance, bearing, x, y) )
    return result



def assign_cylinders(cylinders, robot_pose, scanner_displacement, 
                     reference_cylinders):
    scanner_pose = (robot_pose[0] + cos(robot_pose[2]) * scanner_displacement,
                    robot_pose[1] + sin(robot_pose[2]) * scanner_displacement,
                    robot_pose[2])
    result = []
    for c in cylinders:
        x, y = LegoLogfile.scanner_to_world(scanner_pose, c[2:4])
        best_dist_2 = 1e300
        best_ref = None
        for ref in reference_cylinders:
            dx, dy = ref[0] - x, ref[1] - y
            dist_2 = dx * dx + dy * dy
            if dist_2 < best_dist_2:
                best_dist_2 = dist_2
                best_ref = ref
        if best_ref:
            result.append((c[0:2], best_ref))

    return result
    
    
    
def write_error_ellipses(file_desc, line_header, error_ellipse_list):
    file_desc.writelines( line_header)
    for e in error_ellipse_list:
        file_desc.writelines( "%.3f %.1f %.1f" % e)
    file_desc.writelines( "\n" )
    


def write_error_ellipses(file_desc, line_header, covariance_matrix_list): 
    file_desc.writelines(line_header)
    for m in covariance_matrix_list:
        eigenvals, eigenvects = np.linalg.eig(m)
        angle = atan2(eigenvects[1,0], eigenvects[0,0])
        file_desc.writelines("%.3f %.1f %.1f" % \
                 (angle, sqrt(eigenvals[0]), sqrt(eigenvals[1])))
    file_desc.writelines("\n")
    
    

def find_cylinders(scan, scan_derivative, jump, min_dist):
    cylinder_list = []
    on_cylinder = False
    sum_ray, sum_depth, rays = 0.0, 0.0, 0

    for i in range(len(scan_derivative)):
        if scan_derivative[i] < -jump:
            on_cylinder = True
            sum_ray, sum_depth, rays = 0.0, 0.0, 0
        elif scan_derivative[i] > jump:
            if on_cylinder and rays:
                cylinder_list.append((sum_ray/rays, sum_depth/rays))
            on_cylinder = False
        elif scan[i] > min_dist:
            sum_ray += i
            sum_depth += scan[i]
            rays += 1
    return cylinder_list



def get_cylinders_from_scan(scan, jump, min_dist, cylinder_offset): 
    der = compute_derivative(scan, min_dist)
    cylinders = find_cylinders(scan, der, jump, min_dist)
    result = []
    for c in cylinders:
        bearing = LegoLogfile.beam_index_to_angle(c[0])
        distance = c[1] + cylinder_offset
        x, y = distance*cos(bearing), distance*sin(bearing)
        result.append( (np.array([distance, bearing]), np.array([x, y])) )
    return result
    
    

def get_mean(particles): 
    mean_x, mean_y = 0.0, 0.0
    head_x, head_y = 0.0, 0.0
    for p in particles:
        x, y, theta = p.pose
        mean_x += x
        mean_y += y
        head_x += cos(theta)
        head_y += sin(theta)
    n = max(1, len(particles))
    return np.array([mean_x / n, mean_y / n, atan2(head_y, head_x)])



def get_error_ellipse_and_heading_variance(particles, mean): 
    center_x, center_y, center_heading = mean
    n = len(particles)
    if n < 2:
        return (0.0, 0.0, 0.0, 0.0)

    sxx, sxy, syy = 0.0, 0.0, 0.0
    for p in particles:
        x, y, theta = p.pose
        dx = x - center_x
        dy = y - center_y
        sxx += dx * dx
        sxy += dx * dy
        syy += dy * dy
    cov_xy = np.array([[sxx, sxy], [sxy, syy]]) / (n-1)

    var_heading = 0.0
    for p in particles:
        dh = (p.pose[2] - center_heading + pi) % (2*pi) - pi
        var_heading += dh * dh
    var_heading = var_heading / (n-1)

    eigenvals, eigenvects = np.linalg.eig(cov_xy)
    ellipse_angle = atan2(eigenvects[1,0], eigenvects[0,0])

    return (ellipse_angle, sqrt(abs(eigenvals[0])),
            sqrt(abs(eigenvals[1])),
            sqrt(var_heading))



def print_particles(particles, file_desc): 
    if not particles:
        return    
    file_desc.writelines("PA ")
    for p in particles:
        file_desc.writelines("%.0f %.0f %.3f" % tuple(p.pose))
    file_desc.writelines("\n")
    
    

def write_cylinders(file_desc, line_header, cylinder_list):
    file_desc.writelines(line_header)
    for c in cylinder_list:
        file_desc.writelines("%.1f %.1f" % c)
    file_desc.writelines("\n")



def compute_derivative(scan, min_dist):
    jumps = [ 0 ]
    for i in range(1, len(scan) - 1):
        l = scan[i-1]
        r = scan[i+1]
        if l > min_dist and r > min_dist:
            derivative = (r - l) / 2.0
            jumps.append(derivative)
        else:
            jumps.append(0)
    jumps.append(0)
    return jumps



def find_cylinders(scan, scan_derivative, jump, min_dist): #y
    cylinder_list = []
    on_cylinder = False
    sum_ray, sum_depth, rays = 0.0, 0.0, 0

    for i in range(len(scan_derivative)):
        if scan_derivative[i] < -jump:
            on_cylinder = True
            sum_ray, sum_depth, rays = 0.0, 0.0, 0
        elif scan_derivative[i] > jump:
            if on_cylinder and rays:
                cylinder_list.append((sum_ray/rays, sum_depth/rays))
            on_cylinder = False
        elif scan[i] > min_dist:
            sum_ray += i
            sum_depth += scan[i]
            rays += 1
    return cylinder_list



def get_observations(scan, jump, min_dist, cylinder_offset, #y
                     robot_pose, scanner_displacement,
                     reference_cylinders, max_reference_distance):
    der = compute_derivative(scan, min_dist)
    cylinders = find_cylinders(scan, der, jump, min_dist)
    scanner_pose = (robot_pose[0] + cos(robot_pose[2]) * scanner_displacement,
                    robot_pose[1] + sin(robot_pose[2]) * scanner_displacement,
                    robot_pose[2])
    result = []
    for c in cylinders:
        angle = LegoLogfile.beam_index_to_angle(c[0])
        distance = c[1] + cylinder_offset
        x, y = distance*cos(angle), distance*sin(angle)
        x, y = LegoLogfile.scanner_to_world(scanner_pose, (x, y))
        best_dist_2 = max_reference_distance * max_reference_distance
        best_ref = None
        for ref in reference_cylinders:
            dx, dy = ref[0] - x, ref[1] - y
            dist_2 = dx * dx + dy * dy
            if dist_2 < best_dist_2:
                best_dist_2 = dist_2
                best_ref = ref
        if best_ref:
            result.append(((distance, angle), best_ref))
    return result
    

