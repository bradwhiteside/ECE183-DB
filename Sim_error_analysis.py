import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.fft import fft, ifft

from Plot_results import plot_all_results


# true_data = np.loadtxt('error_analysis/true_data.txt')
# est_data = np.loadtxt('error_analysis/est_data.txt')

# error = np.loadtxt('error_analysis/errors.txt')
error = np.loadtxt('error_analysis/Testerrors((19, 24), (211, 20)).txt')
times = np.loadtxt('error_analysis/times.txt')
true_data = np.loadtxt('error_analysis/true_data.txt')
est_data = np.loadtxt('error_analysis/est_data.txt')
input_goal = np.loadtxt('error_analysis/input_goal.txt')
yaw_goal = np.loadtxt('error_analysis/yaw_goal.txt')





# sqr_diff = np.sqrt(true_states**2 - est_states**2)
# print("difference shape is:",difference.shape)
state_mean = np.mean(error, axis =0)
state_std = np.std(error, axis=0)
print("mean shape is:",state_mean.shape)
print("std shape is:", state_std.shape)


bin_num = 30

fig1, ax1 = plt.subplots(6,1, figsize=(10,  7))
fig1.suptitle('Normal Distribution of Outputs', fontsize=16)
domain = np.linspace(-3,3,100)
# ax1[0].plot(domain, norm.pdf(domain,state_mean[0], state_std[0]), label = "normal dist_x")
ax1[0].hist(error[:,0], bins=bin_num, density = True, label = "x_dist")
ax1[0].plot(domain, norm.pdf(domain,state_mean[0], state_std[0]), label = "normal x_dist")
ax1[0].set_xlabel('m')
# ax1[0].set_ylabel('')
ax1[0].legend()


domain = np.linspace(-3,3,100)
ax1[1].hist(error[:,1], bins=bin_num, density = True, label = "y_dist")
ax1[1].plot(domain, norm.pdf(domain,state_mean[1], state_std[1]), label = "normal y_dist")
ax1[1].set_xlabel('m')
# ax1[1].set_ylabel('')
ax1[1].legend()

domain = np.linspace(-3,3,100)
ax1[2].hist(error[:,2], bins=bin_num, density = True, )
ax1[2].plot(domain, norm.pdf(domain,state_mean[2], state_std[2]), label = "normal z_dist")
ax1[2].set_xlabel('m')
ax1[2].set_ylabel('')
ax1[2].legend()


#Angles
domain = np.linspace(-5,5,100)

ax1[3].hist(np.degrees(error[:,6]), bins=bin_num, density = True, label = "roll_dist")
# ax1[3].plot(domain, norm.pdf(domain, np.degrees(state_mean[6]), np.degrees(state_std[6])), label = "normal dist_roll")
ax1[3].set_xlabel('deg')
# ax1[3].set_ylabel('')
ax1[3].set_xlim([-5,5])
ax1[3].legend()

ax1[4].hist(np.degrees(error[:,7]), bins=bin_num, density = True, label = "pitch_dist")
# ax1[4].plot(domain, norm.pdf(domain,np.degrees(state_mean[7]), np.degrees(state_std[7])), label = "normal dist_pitch")
ax1[4].set_xlabel('deg')
ax1[4].set_xlim([-5,5])
# ax1[4].set_ylabel('')
ax1[4].legend()

ax1[5].hist(np.degrees(error[:,8]), density = True, label = "yaw_dist")
# ax1[4].plot(domain, norm.pdf(domain,np.degrees(state_mean[7]), np.degrees(state_std[7])), label = "normal dist_pitch")
ax1[5].set_xlabel('deg')
# ax1[4].set_ylabel('')
ax1[5].set_xlim([-1,1])
ax1[5].legend()  
ax1[5].set_xlim([-5,5])
# # plt.show()
plt.tight_layout(h_pad = 0.7)



#Frequency analysis
fig2, ax2 = plt.subplots(6,1, figsize=(7,  7))
fig2.suptitle('Amplitude Plot', fontsize=16)

time_length = times[-1]  #Time length
number_of_samples = len(times) #Sample numbers
samplig_freq = number_of_samples/time_length

freq_resolution = samplig_freq/number_of_samples

print("sampling freq {}".format(samplig_freq))
print("resolution:{}".format(freq_resolution))

Nyquist = samplig_freq/2
freq = np.arange(0, Nyquist, freq_resolution)
print("Freq size:", len(freq))
print("Nyquist freq:", Nyquist)


labels =["x_axis", "y_axis", "z_axis", "Roll axis", "Pitch axis", "Yaw axis"]

for i in range(len(labels)):
	j = i
	if i>2:
		j = i + 3

	y = fft(true_data[:,j])
	num_points = int(len(y)/2)
	y = 2 * (y[:num_points])

	mid_inx = int(len(freq)/50)
	ax2[i].bar(freq[:mid_inx], abs(y[:mid_inx])/number_of_samples, width = 0.01)
	ax2[i].set_ylabel('{} Mag'.format(labels[i]))
	ax2[i].set_xlabel('freq (Hz)')
	# # plt.plot(np.angle(y), 'o')

plt.tight_layout(h_pad = 0.7)#, w_pad)
plt.show()
# plot_all_results(times = times, true_states = true_data, est_states = est_data, input_goal= input_goal, yaw_goal =yaw_goal, plt_show=True)


#appy 6 sigma rule
# divitation = state_std *3
# print(divitation)


#intrapolate goal
#see how much it has followed the goal path
