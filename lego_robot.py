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
                    self.scan_data.append(tuple( list(map(int, sp[3:]))))
                else:
                    self.scan_data.append(tuple( list(map(int, sp[2:]))))

            
            elif sp[0] == 'I':
                if first_pole_indices:
                    self.pole_indices = []
                    first_pole_indices = False
                self.pole_indices.append(tuple( list(map(int, sp[2:]))))

            
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
                self.filtered_positions.append( tuple( list(map(float, sp[1:]))) )

            
            elif sp[0] == 'E':
                if first_filtered_stddev:
                    self.filtered_stddev = []
                    first_filtered_stddev = False
                self.filtered_stddev.append( tuple( list(map(float, sp[1:]))) )
                
            
            elif sp[0] == 'L':
                if first_landmarks:
                    self.landmarks = []
                    first_landmarks = False
                if sp[1] == 'C':
                    self.landmarks.append( tuple(['C'] + list(map(float, sp[2:]))) )
                    
            
            elif sp[0] == 'D':
                if sp[1] == 'C':
                    if first_detected_cylinders:
                        self.detected_cylinders = []
                        first_detected_cylinders = False
                    cyl = list(map(float, sp[2:]))
                    self.detected_cylinders.append([(cyl[2*i], cyl[2*i+1]) for i in range(len(cyl)/2)])

            
            elif sp[0] == 'W':
                if sp[1] == 'C':
                    if first_world_cylinders:
                        self.world_cylinders = []
                        first_world_cylinders = False
                    cyl = list(map(float, sp[2:]))
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
