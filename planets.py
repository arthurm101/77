'''
****************************************************************************
****************************************************************************
****************************************************************************
Physics 77 Capstone Project: SIMULATING BINARY ORBITS **********************
Arthur Martirosyan *********************************************************
Sofie Seilnacht*************************************************************
Oscar Antonio Chavez Ortiz *************************************************
****************************************************************************
****************************************************************************
****************************************************************************
'''
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii

#assigning newtons gravitational constant to the value below
G = 6.674e-11

#making a solar mass variable
M_sun = 2e30 #kg

#luminosity of the sun in watts
L_sun = 3.827e26 #watts

#Solar radius in meters
R_sun = 6.957e8 #m

#Stefan-Boltzmann constant
stef_boltz = 5.67e-8

#Surface temperature of the Sun
T_sun = 5778 #K

# Astronomical unit in metres
AU = (149.6e6 * 1000)


#code that reads in the data from the files
data = ascii.read('table1.dat', readme = 'ReadMe')
data1 = ascii.read('table3.dat', readme = 'ReadMe')
data2 = ascii.read('table5.dat', readme = 'ReadMe')

#########
#OUTLINE OF CODE:
#first step we need to find the period of the binary system
#we do this by plotting the radial velocities and then we will zoom in on
#a region and see if we can deduce the period from it.
#once we have the period all we need next will be to find out the mass and average
#separation
#we can find ratio of mass by taking the ratio of radial velocities
##########


#assigning the values of radial velocity to these variables below
#as well as time, one for data set 1
#and another one for data set 2

bvmag = data['B-V']

bvmag1 = bvmag[0]
bvmag2 = bvmag[1]
bvmag3 = bvmag[2]

#this code block here just gathers the radial velocities from data set 1

#data set 1
rad_vel1 = data1['RV1']
rad_vel2 = data1['RV2']
time = data1['HJD']

#this code block here just gathers the radial velocities from data set 2 but we did not end up using it

#data set 2
rad_vel11 = data2['RV1']
rad_vel22 = data2['RV2']
time1 = data2['HJD']

#the code below will index our radial velocity so that we can zoom in on a region of interest
#in this case this will find for data set 1 the radial velocity of RV1 from when
#time is less than 54500

#initializer variable
i = 0

#list that will hold the index
time_segment = []

#while loop that loops through the condition required, in this case that time[i] < 54500
while i < len(time):
	if time[i] < 54500:
		time_segment.append(i)
	i+=1

#assigning the value of index to the length of the list, this will be used below to
#only get the time and radial velocity values that we want
index = len(time_segment)


#code that cuts away unwanted values and only focuses on the region of interest
rv1_segment = rad_vel1[:index]
time_adjusted = time[:index]


#this code is an exact copy of the above just that now we are looking at a different range
#and applying it to a different variable rad_vel2

#initializer variable
i = 0

#list that will hold the index
time_segment1 = []

#while loop that loops through the condition required in this case that 56000< time[i] < 57000
while i < len(time):
	if time[i] > 56000 and time[i] < 57000:
		time_segment1.append(i)
	i+=1

#this code here assigns to a variable where we will start in the list splicing and where we will end
#what we are looking for is something like rad_vel[start:end]
#so we need to know where to start and when to end

#code that assigns the starting and end index below
start = time_segment1[0]
end = time_segment1[-1]


#list that will hold the indexes for the zoomed in region for radial velocity 1
zoomRV1 = []

#initializing variable for loop
i=0

#finding the index for when 56200 < time[i] < 56260 and appending that to zoomRV1
while i < len(time):
	if time[i] > 56200 and time[i] < 56260:
		zoomRV1.append(i)
	i+=1

#assigning start and end splicing values for when we zoom in
zoomstart = zoomRV1[0]
zoomend = zoomRV1[-1]

#the zoomed in rad_vel and time are spliced below
#our first zoomed in value
RV1_zoom = rad_vel1[zoomstart : zoomend]
timezoom = time[zoomstart : zoomend]

#the code below will mimic what we did above but now for radial velocity 2
zoomRV2 = []

#initializing indexing variable
i=0

#while loop that appends the index for when 56200<time[i]<56300
while i < len(time):
	if time[i] > 56200 and time[i] < 56300:
		zoomRV2.append(i)
	i+=1

#assigning our starting and ending variables that will zoom in on the region for rad_vel2
zoomstart2 = zoomRV2[0]
zoomend2 = zoomRV2[-1]

#code that actually takes the piece we are interested in.
RV2_zoom = rad_vel2[zoomstart2 : zoomend2]
tprime = time[zoomstart2 : zoomend2]

#code that zooms in on rv1,rv2 and time using the previous start and end values
rv2_segment = rad_vel2[start:end]
rv1_segment2 = rad_vel1[start:end]
time_seg = time[start:end]

#this code block plots the entire data from data set 1
plt.title('Plot of the entire data set for data1')
plt.plot(time, rad_vel1,'.', label = 'Radial Velocity 1', color = 'blue')
plt.plot(time, rad_vel2,'.', label = 'Radial Velocity 2', color = 'red')
plt.xlabel('Time (days)')
plt.ylabel('Radial Velocity (km/s)')
plt.legend()
plt.savefig('Dataset1.png')
plt.show()


#code that plots the zoomed in region for radial velocity 1
plt.figure()
plt.title('Zoomed in region for RV1')
plt.scatter(time_adjusted, rv1_segment, c = 'blue')
plt.xlabel('Time (days)')
plt.ylabel('Radial Velocity (km/s)')
plt.show()

plt.figure()

plt.title('Different Zoomed in region for RV1')
plt.scatter(time_seg, rv1_segment2, c = 'blue')
plt.xlabel('Time (days)')
plt.ylabel('Radial Velocity (km/s)')
plt.savefig('Radvel1.png')
plt.show()


plt.figure()

#code that plots the zoomed in region for radial velocity 1
plt.title('Total Zoomed in region for RV1')
plt.scatter(timezoom, RV1_zoom, c = 'blue')
plt.xlabel('Time (days)')
plt.ylabel('Radial Velocity (km/s)')
plt.savefig('Rad_vel1.png')
plt.show()

plt.figure()

plt.title('Zoomed in region for RV2')
plt.scatter(time_seg, rv2_segment, c = 'red')
plt.xlabel('Time (days)')
plt.ylabel('Radial Velocity (km/s)')
plt.savefig('Radvel2.png')
plt.show()

plt.figure()

plt.title('Total Zoomed in region for RV2')
plt.scatter(tprime, RV2_zoom, c = 'red')
plt.xlabel('Time (days)')
plt.ylabel('Radial Velocity (km/s)')
plt.xlim(min(tprime), max(tprime))
plt.savefig('Rad_vel2.png')
plt.show()

#from this plot and other zoomed in regions we can calculate the orbital period in days by
#looking for one period of a sine wave


plt.figure()


#Curve fitting for RV1
x_data = timezoom -56200
y_data = RV1_zoom
x = np.linspace(0,20*np.pi, 43)
y = 25.6*np.cos(.191*x)+21
plt.title('Curve fit for Zoomed in region for RV1')
plt.plot(x_data,y_data)
plt.plot(x, y,label='fit: a = 25.6,b = .191, c = 21.0')
#plt.plot(x, y_data-y)
plt.xlabel('Time (days)')
plt.ylabel('Radial Velocity (km/s)')
plt.legend(loc=4)
plt.show()


#Curve fitting for RV2
x_data = tprime -56200
y_data = RV2_zoom
x = np.linspace(0,25*np.pi)
y = 50*np.sin(.19*x-1.5)+17.1
plt.title('Curve fit for Zoomed in region for RV2')
plt.plot(x_data,y_data)
plt.plot(x, y,label='fit: a = 50.1,b = .190, c = 17.1')
plt.xlabel('Time (days)')
plt.ylabel('Radial Velocity (km/s)')
plt.legend(loc=4)
plt.show()


######
#this code block plots the entirety of the data from data set 2
#but we did not end up using this
#it was here just in case data set 1 did not yield good graphs
######

#plt.title('Data from dataset 2')
#plt.scatter(time1, rad_vel11, c = 'blue')
#plt.scatter(time1, rad_vel22, c = 'red')
#plt.xlabel('Time (days)')
#plt.ylabel('Radial Velocity')
#plt.show()


########
#from the plots above we are able to infer that the orbital period for star 1 is of order
#35 days +/- 2 days
#and for the second object/star, we can infer that the period is about 35 days as well.
########


####
#code blocks that calculate astrophysical quantities of interest.
#these are going to include mass, period, semi-major axis
####

#converts the period into seconds which we need for keplars third law
period = 35 * 86400 #in seconds

#space in the output display
print()


#function that calculates the distance away from center of mass
#here we are assuming circular orbits and we use the fact that
#v = distance/period = 2piD/Period
#solving for D we get
#D = v*Period/(2*pi)

def separation(velocity, period):
    rCOM = (period * velocity)/(2 * np.pi)

    return rCOM

#assigning variables to quantities we need

#this first one is velocity of object 1 in meters per second
v1 = 25.6 * 1000 #converst km/s -> m/s

#assigning separation of object 1 from center of mass
a1 = separation(v1, period)

#assigning the value of velocity for object 2 in meters per second
v2 = 52.6 * 1000 #converts km/s->m/s

#assigning the value of separation from center of mass from object 2
a2 = separation(v2, period)

#calculating the total separation
a = a1+a2

#displays the total separation on the screen/display
print('Total separation is: ', a, ' meters or ', a/(1.5e11), ' AU')

#function of keplars third law
#we use this to solve for mass if we are given period and separation
#both of which we now know
def keplar(period, a):
    numerator = (4 * np.pi**2 * a**3)
    denom = G * period**2
    Mtot = numerator/denom
    return Mtot

#assigning the combined mass of the system to variable Mtot
Mtot = keplar(period, a)

#displays it on the screen
print()
print('Total mass of the system is: ', Mtot, 'kg or ', Mtot/M_sun, 'Solar Mass')

#functions that calculate the mass for each individual object
#here since we know the 2 velocities we can get the indivisual masses by taking ratios of velocities
#the exact derivation is seen in the proposal/powerpoint
def Mass2(v1, v2, Mtot):
    M2 = (v1 * Mtot)/ (v1+v2)
    return M2

def Mass1(v1, v2, Mtot):
    M1 = (v2*Mtot)/(v2+v1)
    return M1

#assinging the mass for each individual object to their respective variable
M1 = Mass1(v1, v2, Mtot)
M2 = Mass2(v1, v2, Mtot)

#displaying the results onto the screen
print()
print('Mass of object 1 is: ', M1, 'kg or ', M1/M_sun)
print()
print('Mass of object 2 is: ', M2, 'kg or ', M2/M_sun)

##########
#since both of these stars are of the order a solar mass
#they will most likely have properties like that of the Sun
#meaning that they will have a luminosity roughly equal to the sun
#surface flux and temoerature will also roughly be equal
##########


#The code below just goes and calculates more astrophysical data

#function that calculates the luminosity of the stars given their mass
def luminosity(M):
	#a bunch of if-else statements that converts a given mass range into the appropriate luminosity
	if M < .43*M_sun:
		return .23*(M/M_sun)**(2.3)*L_sun
	elif M > .43*M_sun and M < 2*M_sun:
		return (M/M_sun)**4 * L_sun
	elif M > 2*M_sun and M < 20*M_sun:
		return 1.4 * (M/M_sun)**(3.5) * L_sun
	else:
		return 'Could not calculate luminosity.'

#assigns the luminosity to the assigned variables
lum1 = luminosity(M1)
lum2 = luminosity(M2)

print()
print('Luminosty of star 1 is: ', lum1, 'Watts')
print('Luminosty of star 2 is: ', lum2, 'Watts')
print()


#########
#Another piece of data that we are given is in data table1 we are given the B-V magnitude
#from that we can find the estimated surface temperature assuming that the star radiates like a blackbody
#we find that temperature by using Ballesteros formula
#########

#function that calculates the temperature of the star given its B-V magnitude
def bvtemp(bvmag):

	frac1 = 1/(.92*bvmag + 1.7)
	frac2 = 1/(.92*bvmag + .62)
	T = 4600*(frac1 + frac2)
	return T

temp_bv1 = bvtemp(bvmag1)
temp_bv2 = bvtemp(bvmag2)
temp_bv3 = bvtemp(bvmag3)

#displays the result onto the screen
print()
print('The estimated surface temperature using B-V magnitude is: ')
print('For star 1: ', temp_bv1, 'K')
print('For star 2: ', temp_bv2, 'K')
#print('For star 3: ', temp_bv3, 'K')



########
#Now that we have the temperature and luminosity of the star we can calculate what is its radius is
#using Stefan-Boltzmann law and comparing the stars to the sun
########

#function that calculates the the radius of the star given its temperature
def radii_calc(T,L):
	radii = np.sqrt(L/L_sun) * (T_sun/T)**2 * R_sun
	return radii

#assinging the values for radius to these variables
R_1 = radii_calc(temp_bv1, lum1)
R_2 = radii_calc(temp_bv2, lum2)

#output code
print()
print('The radius for star 1 is: ', R_1, 'm or ', R_1/R_sun, 'Solar Radii')
print('The radius for star 2 is: ', R_2, 'm or ', R_2/R_sun, 'Solar Radii')


########
#End of Graphs
########


'''
BREAK IN CODE ******************************************************************************

'''

########
#This section of code focuses on the Animations
########

import numpy
import math
import pylab
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# One day in seconds (timestep for the simulation)
timestep = 24 * 3600

class planet(object):
    def __init__(self):
        self.px = 0.0
        self.py = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.v = 0.0
        self.mass = 0.0
        self.time = 0.0
        self.energy = 0.0
    def compute_force(self, others):
        self.total_fx = self.total_fy = 0.0
        for other in others:
            # Compute the distance of the other body.
            sx, sy = self.px, self.py
            ox, oy = other.px, other.py
            dx = (ox-sx)
            dy = (oy-sy)
            d = numpy.sqrt(dx ** 2 + dy ** 2)

            # Compute the force of attraction
            f = G * self.mass * other.mass / (d ** 2)

            # Compute the direction of the force.
            theta = math.atan2(dy, dx)
            fx = math.cos(theta) * f
            fy = math.sin(theta) * f

            # Add to the total force exerted on the planet
            self.total_fx += fx
            self.total_fy += fy

    def update_position(self):
        self.vx += self.total_fx / self.mass * timestep
        self.vy += self.total_fy / self.mass * timestep
        self.px += self.vx * timestep
        self.py += self.vy * timestep
        self.time += timestep

    def update_position1(self, others):
        self.vx += self.total_fx / self.mass * timestep
        self.vy += self.total_fy / self.mass * timestep
        self.px += self.vx * timestep
        self.py += self.vy * timestep
        self.v = np.sqrt(self.vx**2 + self.vy**2)
        for other in others:
            other.vx += other.total_fx / other.mass * timestep
            other.vy += other.total_fy / other.mass * timestep
            other.v = np.sqrt(other.vx**2 + other.vy**2)
            self.energy +=  .5 *self.mass * self.v**2 + .5 * other.mass * other.v**2 - 2*(G * self.mass * other.mass)/a
            self.energy = other.energy
        self.time += timestep

values = []
values2 = []

def animate(i, bodies, lines):
    for ind, body in enumerate(bodies):
        body.compute_force(numpy.delete(bodies, ind))
    for body in bodies:
        body.update_position()
    for i in range(len(bodies)):
        lines[i].set_data(bodies[i].time / 3600, bodies[i].vy /1000)
        values.append(bodies[0].vy/1000+21)
        values2.append(.9*bodies[1].vy/1000+17)
    return lines

def animate1(i, bodies, lines):
    for ind, body in enumerate(bodies):
        body.compute_force(numpy.delete(bodies, ind))
    for body in bodies:
        body.update_position1(numpy.delete(bodies, ind))
        #body.energy()
    for i in range(len(bodies)):
        lines[i].set_data(bodies[i].time / 3600, bodies[i].energy/1e+39 )
    return lines

def animate2(i, bodies, lines):
    for ind, body in enumerate(bodies):
        body.compute_force(numpy.delete(bodies, ind))
    for body in bodies:
        body.update_position()
    for i in range(len(bodies)):
        lines[i].set_data(bodies[i].px / AU, bodies[i].py / AU)
    return lines

def main():
    body1 = planet()
    body2 = planet()

	#BODY1
    body1.mass =    M1
    body1.px =  -a1
    body1.vy = v1
    body1.color = 'b'
    body1.size = 10
    body1.name =   'Binary1'

	#BODY2
    body2.mass =   M2
    body2.px =   a2
    body2.vy =   -v2
    body2.color = 'm'
    body2.size = 5
    body2.name =  'Binary2'

    bodies = [ body1, body2]
    lines = [None] * len(bodies)
    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot()

	#CODE TO SET BACKGROUND
    orbit_size = .3
    img = plt.imread("star.png")
    ax.imshow(img, extent=[-orbit_size, orbit_size, -orbit_size, orbit_size])

    for i in range(len(bodies)):
        lines[i], = ax.plot(bodies[i].px / AU, bodies[i].py / AU,
        marker='o', color=bodies[i].color, ms=bodies[i].size,
        label=bodies[i].name)

    ani = animation.FuncAnimation(fig, animate2, numpy.arange(1, 500),
        fargs=[bodies, lines], interval=20, blit=True, repeat=True)

    ax.set_xlabel('x [AU]')
    ax.set_ylabel('y [AU]')

    ax.set_xlim(-orbit_size, orbit_size)
    ax.set_ylim(-orbit_size, orbit_size)

    legend = ax.legend(loc=9, bbox_to_anchor=(0.5, 1.1), ncol=3)
    legend.legendHandles[0]._legmarker.set_markersize(6)

    #PLOT
    plt.show()

if __name__ == "__main__":
    main()


def main():
	#Make obejct instance
    body1 = planet()
    body2 = planet()

	#BODY1
    body1.mass =    M1     #5.9742 * 10 ** 24
    body1.px =  -a1  #-1 * AU
    body1.vy = v1  #29.783 * 1000
    body1.color = 'b'
    body1.size = 10
    body1.name =   'Binary1'

	#BODY2
    body2.mass =   M2    #4.8685 * 10 ** 24
    body2.px =   a2    #0.723 * AU
    body2.vy =   -v2    #-35.02 * 1000
    body2.color = 'w'
    body2.size = 5
    body2.name =  'Binary2'

    bodies = [ body1, body2]
    lines = [None] * len(bodies)
    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot()

	#CODE TO SET BACKGROUND
    orbit_size = .3
    #img = plt.imread("star.png")
    #ax.imshow(img, extent=[-orbit_size, orbit_size, -orbit_size, orbit_size])

	#INITIAL PLOT ALL BODIES
    for i in range(len(bodies)):
        lines[i], = ax.plot(bodies[i].time / 3600, bodies[i].vy /1000 ,marker='o', color=bodies[i].color, ms=bodies[i].size,label=bodies[i].name)

	#UPDATE POSITION ON GRAPH
    ani = animation.FuncAnimation(fig, animate, numpy.arange(1, 20),fargs=[bodies, lines], interval=20, blit=True, repeat=True)
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Radial Velocity [km/s]')
    ax.set_xlim(0, 2000)
    ax.set_ylim(-50, 50)
    legend = ax.legend(loc=9, bbox_to_anchor=(0.5, 1.1), ncol=3)
    legend.legendHandles[0]._legmarker.set_markersize(6)

	#plots our data on top of animation
    plt.scatter((timezoom-56198)*24, RV1_zoom-24, c = 'red')
    plt.show()

if __name__ == "__main__":
    main()


def main():
	#Make obejct instance
    body1 = planet()
    body2 = planet()

	#BODY1
    body1.mass =    M1     #5.9742 * 10 ** 24
    body1.px =  -a1  #-1 * AU
    body1.vy = v1  #29.783 * 1000
    body1.color = 'b'
    body1.size = 10
    body1.name =   'Binary1'

	#BODY2
    body2.mass =   M2    #4.8685 * 10 ** 24
    body2.px =   a2    #0.723 * AU
    body2.vy =   -v2    #-35.02 * 1000
    body2.color = 'm'
    body2.size = 5
    body2.name =  'Binary2'

    bodies = [ body1, body2]
    lines = [None] * len(bodies)
    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot()

	#CODE TO SET BACKGROUND
    orbit_size = .3
    #img = plt.imread("star.png")
    #ax.imshow(img, extent=[-orbit_size, orbit_size, -orbit_size, orbit_size])
    for i in range(len(bodies)):
        lines[i], = ax.plot(bodies[i].time / 3600, bodies[i].energy / 1e+39,
        marker='o', color=bodies[i].color, ms=bodies[i].size,
        label=bodies[i].name)

    ani = animation.FuncAnimation(fig, animate1, numpy.arange(1, 500),
        fargs=[bodies, lines], interval=20, blit=True, repeat=True)

    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Energy [J*1e+39]')
    ax.set_xlim(0, 1500)
    ax.set_ylim(-1000, 0)
    legend = ax.legend(loc=9, bbox_to_anchor=(0.5, 1.1), ncol=3)
    legend.legendHandles[0]._legmarker.set_markersize(6)

    #PLOT
    plt.show()

if __name__ == "__main__":
    main()




########
#Conclusion
########

########
#Here we can compare our modeled data to the actual data recieved
########

plt.title('Data comparison RV1')
plt.plot((timezoom-56198)*24, RV1_zoom, color='b', label = 'Measured Velocity')
hello = np.arange(0,len(values))
print(len(values))
plt.plot(hello*10.9+80,values, color='r', label = 'Simulation Velocity')
plt.xlabel('Time [days]')
plt.ylabel('Radial Velocity [km/s]')
plt.legend(loc=4)
plt.show()

plt.title('Data comparison RV2')
plt.plot((tprime-56198)*24, RV2_zoom, color='b', label = 'Measured Velocity')
hello = np.arange(0,len(values2))
print(len(values2))
plt.plot(hello*10.9+80,values2, color='r', label = 'Simulation Velocity')
plt.xlabel('Time [days]')
plt.ylabel('Radial Velocity [km/s]')
plt.legend(loc=4)
plt.show()

x = np.linspace(1,100)
y = .5*M1*v1**2+.5*M2*v2**2-2*G*M1*M2/x
z = -2*G*M1*M2/x
b = np.ones(50)*.5*M1*v1**2+.5*M2*v2**2
plt.title('Energy versus Radius')
plt.plot(x,y, label='Energy Total')
plt.plot(x,z+y,label='Potential Energy')
plt.plot(x,b,label='Kinetic Energy')
plt.ylim((-1e+50,1e+50))
plt.xlabel('Radius [KM]')
plt.ylabel('Energy [J]')
plt.legend(loc=4)
plt.show()
