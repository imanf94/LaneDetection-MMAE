import os
import airsim
import time
import numpy as np

import re
import cv2
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import math
import lane
import lane2
import adv_lane

def variance(data, ddof=0):
    n = len(data)
    mean = sum(data) / n
    return (sum((x - mean) ** 2 for x in data) / (n - ddof))

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()
i = 0
Kp = 0.001 #0.002 for curvy roads -- 0.001 for straight lane
Ki = 0
Kd = 0.001
offset = []
offset2 = []
main_offs = []
t = []
timestep = []
error = []
error_dot = []
error_int = []
controlledSteering = []
var1 = []
var2 = []
coVar1 = []
coVar2 = []
prob1 = []
prob2 = []
r1 = []
r2 = []
R = 1
T = 1

end_bool = True
P1 = 10
P2 = 10
s1 = 0
s2 = 0
beta1 = 0
beta2 = 0

while end_bool:
    client.simPause(False)
    # get state of the car
    car_state = client.getCarState()
    #print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

    # set the controls for car
    client.simPause(False)
    car_controls.throttle = 0.5
    if(i==0):
        car_controls.steering = 0
    else:
        car_controls.steering = controlledSteering[i-1]

    client.setCarControls(car_controls)

    # let car drive a bit
    time.sleep(0.5)
    #client.simPause(False)
    #client.simContinueForTime(1)

    client.simPause(True)
    #responses = client.simGetImages([airsim.ImageRequest("CAM0", airsim.ImageType.Scene, False, False)])
    responses = client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.Scene, False, False)])

    #responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, True)])
    for response in responses:
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        
        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)
        
        # saving the image in the storage
        fileName = time.strftime("%Y%m%d-%H%M%S")
        dir_path = os.path.dirname(os.path.realpath(__file__))
        airsim.write_png(dir_path + '\\Images\\' + fileName + '.png', img_rgb) 

        # processing the image to find lanes and offset
        offset.append(0) 
        offset2.append(0) 
        main_offs.append(0)
        t.append(0)
        timestep.append(0)
        error.append(0)
        error_dot.append(0)
        error_int.append(0)
        controlledSteering.append(0)
        var1.append(0)
        var2.append(0)
        coVar1.append(0)
        coVar2.append(0)
        r1.append(0)
        r2.append(0)
        prob1.append(0)
        prob2.append(0)

        try:
            offset2[i] = lane.main(img_rgb) #offset (cm-not scaled)
            offset[i] = lane2.main(img_rgb)
            #offset[i] = adv_lane.lane_finding_pipeline(img_rgb)
            
            
        except:
            print("error in image processing")

        # Probability Calc
        epsilon = 0.001 # To keep the posiblities alive
        if (i>=1):
            r1[i] = abs(var1[i-1]-abs(offset[i])**2)+epsilon #residuals
            r2[i] = abs(var2[i-1]-abs(offset2[i])**2)+epsilon
            
            s1=P1+R
            beta1=1 #(((2*math.pi)**1)*(s1))**-0.5
            s2=P2+R
            beta2=1 #(((2*math.pi)**1)*(s2))**-0.5
            #prob[i] = c2/(c1+c2)
            #prob2[i] = c1/(c1+c2)
            prob1[i]=((beta1*(math.exp(-0.5*(r1[i]))*s1**-1*(r1[i])))*prob1[i-1])/((beta1*(math.exp(-0.5*(r1[i]))*s1**-1*(r1[i])))*prob1[i-1]+(beta2*(math.exp(-0.5*(r2[i]))*s2**-1*(r2[i])))*prob2[i-1])
            prob2[i]=((beta2*(math.exp(-0.5*(r2[i]))*s2**-1*(r2[i])))*prob2[i-1])/((beta2*(math.exp(-0.5*(r2[i]))*s2**-1*(r2[i])))*prob2[i-1]+(beta1*(math.exp(-0.5*(r1[i]))*s1**-1*(r1[i])))*prob1[i-1])
        else:
            prob1[0] = 0.5
            prob2[0] = 0.5
            var1[0] = 50
            var2[0] = 50

        main_offs[i] = prob1[i]*offset[i] + prob2[i]*offset2[i]

        #derivative and integral calculations:
        t[i] = round(time.time()*1000) #time in millisecond
        dt = 1
        

        if(i>1):
            error[i] = main_offs[i]
            error_dot[i] = (main_offs[i]-main_offs[i-1])/dt
            error_int[i] = error_int[i-1]+(error[i]+error[i-1])/2*dt
            timestep[i] = timestep[i-1] + dt
            var1[i] = variance(offset)
            var2[i] = variance(offset2)
            coVar1[i] = (offset[i]-offset[i-1])
            coVar2[i] = (offset2[i]-offset2[i-1])
            P1 = coVar1[i]
            P2 = coVar2[i]

        elif(i==1):
            error[i] = main_offs[i]
            error_dot[i] = (main_offs[i]-main_offs[i-1])/dt
            error_int[i] = error_int[i-1]+(error[i]+error[i-1])/2*dt
        elif(i==0):
            error[i] = main_offs[i]
            error_dot[i] = 0
            error_int[i] = 0

        controlledSteering[i] = -(Kp*error[i]+Kd*error_dot[i]+Ki*error_int[i])
        print(error[i],error_dot[i],error_int[i])
        print(controlledSteering[i])
        print()
        i += 1
        
        if (i==40):
            end_bool = False
    
# Plotting the results
client.simPause(True)

# here we are creating sub plots
figure, axes = plt.subplots(3)
line1 = axes[0].plot(timestep, offset, label = "Model 1")
line2 = axes[0].plot(timestep, offset2, label = "Model 2")
line3 = axes[0].plot(timestep, main_offs, label = "Final Model")
axes[0].legend()
axes[0].set(ylabel='Offset (cm)')

line4 = axes[1].plot(timestep, var1, label = "Model 1")
line5 = axes[1].plot(timestep, var2, label = "Model 2")
#plt.title("Offset from center line", fontsize=20)
#plt.xlabel("Time Step")
#plt.ylabel("Standard Deviation")
axes[1].set(ylabel='Standard Deviation', xlabel="Time Step")

line6 = axes[2].plot(timestep, prob1, label = "Model 1")
line7 = axes[2].plot(timestep, prob2, label = "Model 2")
axes[2].set(ylabel='Probability', xlabel="Time Step")

plt.show()



        


        

        

    
