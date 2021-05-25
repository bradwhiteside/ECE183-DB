import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# true_data = np.loadtxt('error_analysis/true_data.txt')
# est_data = np.loadtxt('error_analysis/est_data.txt')

error = np.loadtxt('error_analysis/errors.txt')
# error = true_data - est_data



# sqr_diff = np.sqrt(true_states**2 - est_states**2)
# print("difference shape is:",difference.shape)
state_mean = np.mean(error, axis =0)
state_std = np.std(error, axis=0)
print("mean shape is:",state_mean.shape)
print("std shape is:", state_std.shape)


bin_num = 10

fig1, ax1 = plt.subplots(5,1, figsize=(10,  7))
fig1.suptitle('Normal Distribution of Outputs', fontsize=16)
domain = np.linspace(-1,1,100)
# ax1[0].plot(domain, norm.pdf(domain,state_mean[0], state_std[0]), label = "normal dist_x")
ax1[0].hist(error[:,0], bins=bin_num, density = True, label = "x_dist")
ax1[0].plot(domain, norm.pdf(domain,state_mean[0], state_std[0]), label = "normal x_dist")
ax1[0].set_xlabel('m')
# ax1[0].set_ylabel('')
ax1[0].legend()

domain = np.linspace(-1,1,100)
ax1[1].hist(error[:,1], bins=bin_num, density = True, label = "y_dist")
ax1[1].plot(domain, norm.pdf(domain,state_mean[1], state_std[1]), label = "normal y_dist")
ax1[1].set_xlabel('m')
# ax1[1].set_ylabel('')
ax1[1].legend()

domain = np.linspace(-0.1,0.1,100)
ax1[2].hist(error[:,2], bins=bin_num, density = True, )
ax1[2].plot(domain, norm.pdf(domain,state_mean[2], state_std[2]), label = "normal z_dist")
ax1[2].set_xlabel('m')
# ax1[2].set_ylabel('')
ax1[2].legend()


#Angles
domain = np.linspace(-0.2,0,100)

ax1[3].hist(np.degrees(error[:,6]), bins=bin_num, density = True, label = "roll_dist")
# ax1[3].plot(domain, norm.pdf(domain, np.degrees(state_mean[6]), np.degrees(state_std[6])), label = "normal dist_roll")
ax1[3].set_xlabel('deg')
# ax1[3].set_ylabel('')
ax1[3].legend()

ax1[4].hist(np.degrees(error[:,7]), bins=bin_num, density = True, label = "pitch_dist")
# ax1[4].plot(domain, norm.pdf(domain,np.degrees(state_mean[7]), np.degrees(state_std[7])), label = "normal dist_pitch")
ax1[4].set_xlabel('deg')
# ax1[4].set_ylabel('')
ax1[4].legend()

    
plt.show()

#appy 6 sigma rule
# divitation = state_std *3
# print(divitation)


#intrapolate goal
#see how much it has followed the goal path
