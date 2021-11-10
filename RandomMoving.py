import os
import airsim
import time
import numpy as np
from numpy.core.fromnumeric import size
from scipy.interpolate import InterpolatedUnivariateSpline

import re
import cv2
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import lane
import lane2
import math
import adv_lane

def std(data, ddof=0):
    n = len(data)
    mean = sum(data) / n
    return math.sqrt(sum((x - mean) ** 2 for x in data) / (n - ddof))

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
var = []
var2 = []
var_obs = []
prob = []
prob2 = []
observer = []
cor_obs = []

end_bool = True
while end_bool:
    client.simPause(False)
    # get state of the car
    car_state = client.getCarState()
    #print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

    # set the controls for car
    #client.simPause(False)
    car_controls.throttle = 0.5
    if(i==0):
        car_controls.steering = 0
    else:
        car_controls.steering = controlledSteering[i-1]

    client.setCarControls(car_controls)

    # let car drive a bit
    time.sleep(0.5)
    #client.simContinueForTime(1)
    
    client.simPause(True)
    responses = client.simGetImages([airsim.ImageRequest("CAM0", airsim.ImageType.Scene, False, False),airsim.ImageRequest("CAM1", airsim.ImageType.Scene, False, False)])
    #back_resps = client.simGetImages([airsim.ImageRequest("CAM1", airsim.ImageType.Scene, False, False)])
    #back_data = back_resps[0]
    
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
    var.append(0)
    var2.append(0)
    var_obs.append(0)
    prob.append(0)
    prob2.append(0)
    observer.append(0)

    #responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, True)])
    for response in responses:
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        #back1d =  np.fromstring(back_data.image_data_uint8, dtype=np.uint8) 

        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)
        #back_rgb = back1d.reshape(back_data.height, back_data.width, 3)
        
        # saving the image in the storage
        fileName = time.strftime("%Y%m%d-%H%M%S")
        dir_path = os.path.dirname(os.path.realpath(__file__))
        airsim.write_png(dir_path + '\\Images\\' + fileName + '.png', img_rgb) 
        plt.imshow(img_rgb)
        plt.show()
        

        try:
            if(response.camera_name=='CAM0'): #Front Camera            
                offset2[i] = lane.main(img_rgb) #offset (cm-not scaled)
                offset[i] = lane2.main(img_rgb)
                #offset[i] = adv_lane.lane_finding_pipeline(img_rgb)
            elif(response.camera_name=='CAM1'): #Back Camera
                observer[i] = -lane2.main(img_rgb)
        except:
            print("error in image processing")

    # Probability Calc
    if (i>=1):
        #If the observer offset changes more than 50cm in one timestep -> estimation
        if(abs(cor_obs[i-1]-observer[i])>50):
            s = InterpolatedUnivariateSpline(np.array([1,2]), cor_obs[-3:-2], k=1) #First order extrapolation
            y = s(3) #Third timpstep
            cor_obs[i] = y
        else:
            cor_obs[i] = observer[i]
        try:
            c1 = math.exp(observer[i-1]-(offset[i]))
            c2 = math.exp(observer[i-1]-(offset2[i]))
        except:
            c1 = 0.5
            c2 = 0.5
        prob[i] = c2/(c1+c2)
        prob2[i] = c1/(c1+c2)
        prob[i] = (prob[i]+prob[i-1])/2
        prob2[i] = (prob2[i]+prob2[i-1])/2
        

    else:
        prob[i] = 0.5
        prob2[i] = 0.5

    main_offs[i] = prob[i]*offset[i] + prob2[i]*offset2[i]

    #derivative and integral calculations:
    t[i] = round(time.time()*1000) #time in millisecond
    dt = 1
    

    if(i>1):
        error[i] = main_offs[i]
        error_dot[i] = (main_offs[i]-main_offs[i-1])/dt
        error_int[i] = error_int[i-1]+(error[i]+error[i-1])/2*dt
        timestep[i] = timestep[i-1] + dt
        var[i] = std(offset)
        var2[i] = std(offset2)
        var_obs = std(observer)
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

    if (i==80):
        end_bool = False
    i += 1
# Plotting the results
client.simPause(True)

# here we are creating sub plots
figure, axes = plt.subplots(3)
line1 = axes[0].plot(timestep, offset, label = "Model 1")
line2 = axes[0].plot(timestep, offset2, label = "Model 2")
line3 = axes[0].plot(timestep, main_offs, label = "Final Model")
line04 = axes[0].plot(timestep, observer, label = "Observer")
line05 = axes[0].plot(timestep, observer, label = "Corrected Observer")
axes[0].legend(bbox_to_anchor=(0.5, 1.05), loc="upper center")
axes[0].set(ylabel='Offset (cm)')
axes[0].set_xlim([0, 80])
axes[0].set_ylim([-100, 80])
axes[0].set(size=[10,2])

line4 = axes[1].plot(timestep, var, label = "Model 1")
line5 = axes[1].plot(timestep, var2, label = "Model 2")
#plt.title("Offset from center line", fontsize=20)
plt.xlabel("Time Step")
plt.ylabel("Standard Deviation")
axes[1].set(ylabel='Standard Deviation', xlabel="Time Step")
axes[1].set_xlim([0, 80])
axes[1].set_ylim([0, 80])

line6 = axes[2].plot(timestep, prob, label = "Model 1")
line7 = axes[2].plot(timestep, prob2, label = "Model 2")
axes[2].set(ylabel='Probability', xlabel="Time Step")
axes[2].set_xlim([0, 80])
axes[2].set_ylim([0, 1])

plt.show()



        


        

        

    
