import numpy as np
import math
import scipy.integrate
import time
import datetime
import threading
from controller import Robot, Supervisor



class Propeller():
    def __init__(self, prop_dia, prop_pitch, thrust_unit='N'):
        self.dia = prop_dia
        self.pitch = prop_pitch
        self.thrust_unit = thrust_unit
        self.speed = 0 #RPM
        self.thrust = 0

    def set_speed(self,speed):
        self.speed = speed
        # From http://www.electricrcaircraftguy.com/2013/09/propeller-static-dynamic-thrust-equation.html
        self.thrust = 4.392e-8 * self.speed * math.pow(self.dia,3.5)/(math.sqrt(self.pitch))
        self.thrust = self.thrust*(4.23e-4 * self.speed * self.pitch)
        if self.thrust_unit == 'Kg':
            self.thrust = self.thrust*0.101972


class Quadcopter():
    # State space representation: [x y z x_dot y_dot z_dot theta phi gamma theta_dot phi_dot gamma_dot]
    # From Quadcopter Dynamics, Simulation, and Control by Andrew Gibiansky
    def __init__(self,quads,robot,_super, gravity=9.81, d=0.0245):
        self.quads = quads
        self.g = gravity
        self.d = d #drag factor
        self.thread_object = None
        # self.ode =  scipy.integrate.ode(self.state_dot).set_integrator('vode',nsteps=500,method='bdf')
        self.time = datetime.datetime.now()
        for key in self.quads:
            self.quads[key]['state'] = np.zeros(12)
            self.quads[key]['state'][0:3] = self.quads[key]['position']
            self.quads[key]['state'][6:9] = self.quads[key]['orientation']
            self.quads[key]['m1'] = Propeller(self.quads[key]['prop_size'][0],self.quads[key]['prop_size'][1])
            self.quads[key]['m2'] = Propeller(self.quads[key]['prop_size'][0],self.quads[key]['prop_size'][1])
            self.quads[key]['m3'] = Propeller(self.quads[key]['prop_size'][0],self.quads[key]['prop_size'][1])
            self.quads[key]['m4'] = Propeller(self.quads[key]['prop_size'][0],self.quads[key]['prop_size'][1])
            self.quads[key]['m5'] = Propeller(self.quads[key]['prop_size'][0],self.quads[key]['prop_size'][1])
            self.quads[key]['m6'] = Propeller(self.quads[key]['prop_size'][0],self.quads[key]['prop_size'][1])
            # From Quadrotor Dynamics and Control by Randal Beard
            # ixx=((2*self.quads[key]['weight']*self.quads[key]['r']**2)/5)+(2*self.quads[key]['weight']*self.quads[key]['L']**2)
            # iyy=ixx
            # izz=((2*self.quads[key]['weight']*self.quads[key]['r']**2)/5)+(4*self.quads[key]['weight']*self.quads[key]['L']**2)
            ixx = 0.31  #((2*self.quads[key]['weight']*self.quads[key]['r']**2)/5)+(2*self.quads[key]['weight']*self.quads[key]['L']**2)#0.72#0.2208
            iyy = 0.30 # 0.74#0.2208
            izz = 0.60 #((2*self.quads[key]['weight']*self.quads[key]['r']**2)/5)+(4*self.quads[key]['weight']*self.quads[key]['L']**2)#0.64#0.4386
            print(ixx, iyy, izz)
            self.quads[key]['I'] = np.array([[ixx,0,0],[0,iyy,0],[0,0,izz]])
            self.quads[key]['invI'] = np.linalg.inv(self.quads[key]['I'])

        self.run = True

        #webot setups
        self.hexacpter = _super
        self.drone = self.hexacpter.getFromDef('drone')
        self.webot_timestep = int(self.hexacpter.getBasicTimeStep())
        print("time step is:",self.webot_timestep)

        #Motors
        MAX_SPEED = 310.4
        self.prop1 = self.hexacpter.getDevice("prop1")
        self.prop2 = self.hexacpter.getDevice("prop2")
        self.prop3 = self.hexacpter.getDevice("prop3")
        self.prop4 = self.hexacpter.getDevice("prop4")
        self.prop5 = self.hexacpter.getDevice("prop5")
        self.prop6 = self.hexacpter.getDevice("prop6")
        self.prop1.setPosition(-1*float('inf'))
        self.prop2.setPosition(float('inf'))
        self.prop3.setPosition(-1*float('inf'))
        self.prop4.setPosition(float('inf'))
        self.prop5.setPosition(-1*float('inf'))
        self.prop6.setPosition(float('inf'))
        self.props = [self.prop1, self.prop2, self.prop3, self.prop4, self.prop5, self.prop6]
        self.props[0].setVelocity(0)
        self.props[1].setVelocity(0)
        self.props[2].setVelocity(0)
        self.props[3].setVelocity(0)
        self.props[4].setVelocity(0)
        self.props[5].setVelocity(0)

        
        #Sensors
        self.gyro = self.hexacpter.getDevice('GYRO')
        self.gps = self.hexacpter.getDevice('GPS')
        self.compass = self.hexacpter.getDevice('COMPASS')
        #add acceleromter
 
    def rotation_matrix(self,angles):
        ct = math.cos(angles[0])
        cp = math.cos(angles[1])
        cg = math.cos(angles[2])
        st = math.sin(angles[0])
        sp = math.sin(angles[1])
        sg = math.sin(angles[2])
        R_x = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
        R_y = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        R_z = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R

    
    def wrap_angle(self,val): #converts to -pi to pi
        return( ( val + np.pi) % (2 * np.pi ) - np.pi )

    #For webots true orientation calculations
    def convert_orientation_matrix(self, m):
        sy = m[1, 0];
        cy = m[0, 0]
        yaw = np.arctan2(sy, cy)

        sp = -m[2, 0];
        cp = np.sqrt(m[2, 1] ** 2 + m[2, 2] ** 2)
        pitch = np.arctan2(sp, cp)

        sr = m[2, 1];
        cr = m[2, 2]
        roll = np.arctan2(sr, cr)

        return np.array([roll, pitch, yaw])
        
    def set_motor_speeds(self,quad_name,speeds):
        self.props[0].setVelocity(-speeds[0])
        self.props[1].setVelocity(speeds[1])
        self.props[2].setVelocity(-speeds[2])
        self.props[3].setVelocity(speeds[3])
        self.props[4].setVelocity(-speeds[4])
        self.props[5].setVelocity(speeds[5])
        # print(speeds)
        # self.quads[quad_name]['m1'].set_speed(speeds[0])
        # self.quads[quad_name]['m2'].set_speed(speeds[1])
        # self.quads[quad_name]['m3'].set_speed(speeds[2])
        # self.quads[quad_name]['m4'].set_speed(speeds[3])
        # self.quads[quad_name]['m5'].set_speed(speeds[4])
        # self.quads[quad_name]['m6'].set_speed(speeds[5])


    def get_L(self):
        return self.quads['q1']['L']

    def get_position(self,quad_name):
        return self.quads[quad_name]['state'][0:3]

    def get_linear_rate(self,quad_name):
        return self.quads[quad_name]['state'][3:6]

    def get_orientation(self,quad_name):
        return self.quads[quad_name]['state'][6:9]

    def get_angular_rate(self,quad_name):
        return self.quads[quad_name]['state'][9:12]

    def get_state(self,quad_name):
        # return self.quads[quad_name]['state']
        pos = np.array(self.drone.getPosition())
        lin_v = np.array(self.drone.getVelocity()[:3])
        o = np.array(self.drone.getOrientation()).reshape((3, 3))
        ang = self.convert_orientation_matrix(o.T)
        ang_v = np.array(self.drone.getVelocity()[3:])
        return np.array([pos, lin_v, ang, ang_v]).flatten()
        

    def set_position(self,quad_name,position):
        self.quads[quad_name]['state'][0:3] = position

    def set_orientation(self,quad_name,orientation):
        self.quads[quad_name]['state'][6:9] = orientation

    def get_time(self):
        return self.time

    def thread_run(self,dt,time_scaling):
        rate = time_scaling*dt
        last_update = self.time
        while(self.run==True):
            time.sleep(0)
            self.time = datetime.datetime.now()
            if (self.time-last_update).total_seconds() > rate:
                # self.update(dt)
                self.hexacpter.step(self.webot_timestep)
                last_update = self.time

    def start_thread(self,dt=0.002,time_scaling=1):
        self.thread_object = threading.Thread(target=self.thread_run,args=(dt,time_scaling))
        self.thread_object.start()

    def stop_thread(self):
        self.run = False

    # def state_dot(self, time, state, key):
        # L = self.quads[key]['L']
        # M1 = self.quads[key]['m1'].thrust
        # M2 = self.quads[key]['m2'].thrust
        # M3 = self.quads[key]['m3'].thrust
        # M4 = self.quads[key]['m4'].thrust
        # M5 = self.quads[key]['m5'].thrust
        # M6 = self.quads[key]['m6'].thrust

        # state_dot = np.zeros(12)
        # # The linear velocities(t+1 x_dots equal the t x_dots)   #x_1_dot
        # state_dot[0] = self.quads[key]['state'][3]
        # state_dot[1] = self.quads[key]['state'][4]
        # state_dot[2] = self.quads[key]['state'][5]
        
        # # The acceleration
        # x_dotdot = np.array([0,0,-1*self.g]) + np.dot(self.rotation_matrix( self.quads[key]['state'][6:9]), np.array([0,0,(M1 + M2 + M3 + M4 + M5 + M6)]))/self.quads[key]['weight']  #x_2_dot
        
        # state_dot[3] = x_dotdot[0]
        # state_dot[4] = x_dotdot[1]
        # state_dot[5] = x_dotdot[2]
        # # The angular rates(t+1 theta_dots equal the t theta_dots)  #x_3_dot 
        # state_dot[6] = self.quads[key]['state'][9]
        # state_dot[7] = self.quads[key]['state'][10]
        # state_dot[8] = self.quads[key]['state'][11]
        
        # # The angular accelerations
        # omega = self.quads[key]['state'][9:12]

        # # tau = np.array([self.quads[key]['L']*(self.quads[key]['m1'].thrust-self.quads[key]['m3'].thrust), 
        # #                 self.quads[key]['L']*(self.quads[key]['m2'].thrust-self.quads[key]['m4'].thrust), 
        # #                 self.d*(self.quads[key]['m1'].thrust-self.quads[key]['m2'].thrust+self.quads[key]['m3'].thrust-self.quads[key]['m4'].thrust)])
        # #Torques
        # tau = np.array([L*(-M2+M5+0.5*(-M1-M3+M4+M6)), 
        #                 L*(np.sqrt(3)/2)*(-M1+M3+M4-M6), 
        #                 self.d*(-(self.quads[key]['m1'].speed)**2+(self.quads[key]['m2'].speed)**2-(self.quads[key]['m3'].speed)**2+(self.quads[key]['m4'].speed)**2-(self.quads[key]['m5'].speed)**2+(self.quads[key]['m6'].speed)**2)])



        # x_val = tau[0]
        # y_val = tau[1]
        # z_val = tau[2]
        # throttle = x_dotdot[2]
        # m1 = 1/(6*L) * (L * throttle - 2 * x_val - np.sqrt(3) * y_val - L/self.d *z_val)
        # m2 = 1/(6*L) * (L * throttle -     x_val                      + L/self.d *z_val)
        # m3 = 1/(6*L) * (L * throttle - 2 * x_val + np.sqrt(3) * y_val - L/self.d *z_val)
        # m4 = 1/(6*L) * (L * throttle + 2 * x_val + np.sqrt(3) * y_val + L/self.d *z_val)
        # m5 = 1/(6*L) * (L * throttle +     x_val                      - L/self.d *z_val)
        # m6 = 1/(6*L) * (L * throttle + 2 * x_val - np.sqrt(3) * y_val + L/self.d *z_val)
        # print("%.2f, %.2f, %.2f, %.2f, %.2f, %.2f" % (M1/m1, M2/m2, M3/m3, M4/m4, M5/m5, M6/m6))



        # omega_dot = np.dot(self.quads[key]['invI'], (tau - np.cross(omega, np.dot(self.quads[key]['I'],omega))))
        # state_dot[9] = omega_dot[0]
        # state_dot[10] = omega_dot[1]
        # state_dot[11] = omega_dot[2]
        # return state_dot

    # def update(self, dt):
        # for key in self.quads:
        #     self.ode.set_initial_value(self.quads[key]['state'],0).set_f_params(key)
        #     self.quads[key]['state'] = self.ode.integrate(self.ode.t + dt)
        #     # self.quads[key]['state'] += self.state_dot(1, 1, 'q1') * dt
        #     self.quads[key]['state'][6:9] = self.wrap_angle(self.quads[key]['state'][6:9])
        #     self.quads[key]['state'][2] = max(0,self.quads[key]['state'][2])
        # pass