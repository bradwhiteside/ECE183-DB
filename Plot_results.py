import _tkinter
import matplotlib.pyplot as plt
import numpy as np


def init_plot(plt_show):
    if plt_show == False:
        return None, None, None

    fig1, axes = plt.subplots(3, 2, figsize=(10, 7))
    fig1.suptitle('x, y, z, roll, pitch, yaw', fontsize=16)

    axes[0, 0].set_xlabel('time (s)')
    axes[0, 0].set_ylabel('x (m)')
    #axes[0, 0].legend()

    axes[1, 0].set_xlabel('time (s)')
    axes[1, 0].set_ylabel('y (m)')
    #axes[1, 0].legend()

    axes[2, 0].set_xlabel('time (s)')
    axes[2, 0].set_ylabel('z (m)')
    # axes[2,0].set_ylim([0,11])
    #axes[2, 0].legend()

    axes[0, 1].set_xlabel('time (s)')
    axes[0, 1].set_ylabel('roll (deg)')
    #axes[0, 1].legend()

    axes[1, 1].set_xlabel('time (s)')
    axes[1, 1].set_ylabel('pitch (deg)')
    #axes[1, 1].legend()

    axes[2, 1].set_xlabel('time (s)')
    axes[2, 1].set_ylabel('overshoot (m)')
    #axes[2, 1].legend()
    
    dummy_1D_values = np.array([0])
    dummy_state_values = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((1, 9))
    print(dummy_state_values)
    lines = []

    # axes[0,0].plot(times, est_states[:,0], label = "x dir_est", color = "green")
    lines.extend(axes[0, 0].plot(dummy_1D_values, dummy_state_values[:, 0], label="x goal"))
    lines.extend(axes[0, 0].plot(dummy_1D_values, dummy_state_values[:, 0], label="x dir"))
    lines.extend(axes[0, 0].plot(dummy_1D_values, dummy_state_values[:, 0], label="x est"))

    # axes[1,0].plot(times, est_states[:,1], label = "y dir_est" , color = "green")
    lines.extend(axes[1, 0].plot(dummy_1D_values, dummy_state_values[:, 1], label="y goal"))
    lines.extend(axes[1, 0].plot(dummy_1D_values, dummy_state_values[:, 1], label="y dir"))
    lines.extend(axes[1, 0].plot(dummy_1D_values, dummy_state_values[:, 1], label="y est"))

    # axes[2,0].plot(times, est_states[:,2], label = "z dir_est", color = "green")
    lines.extend(axes[2, 0].plot(dummy_1D_values, dummy_state_values[:, 2], label="z goal"))
    lines.extend(axes[2, 0].plot(dummy_1D_values, dummy_state_values[:, 2], label="z (altitude)"))
    lines.extend(axes[2, 0].plot(dummy_1D_values, dummy_state_values[:, 2], label="z est"))

    # axes[0,1].plot(times, np.degrees(est_states[:,6]), label = "roll_est", color = "green")
    lines.extend(axes[0, 1].plot(dummy_1D_values, dummy_state_values[:, 6], label="roll"))
    lines.extend(axes[0, 1].plot(dummy_1D_values, dummy_state_values[:, 6], label="roll est"))

    # axes[1,1].plot(times, np.degrees(est_states[:,7]), label = "pitch_est", color = "green")
    lines.extend(axes[1, 1].plot(dummy_1D_values, dummy_state_values[:, 7], label="pitch"))
    lines.extend(axes[1, 1].plot(dummy_1D_values, dummy_state_values[:, 7], label="pitch est"))

    # axes[2,1].plot(times, np.degrees(est_states[:,8]), label = "yaw_est", color = "green")
    lines.extend(axes[2, 1].plot(dummy_1D_values, dummy_1D_values, label="overshoot"))

    return fig1, axes, lines


# Plot the path
def plot_results(figure, axes, lines, times, true_states, est_states, torques, speeds, accels, input_goal, overshoots, avg_velocity,
                 plt_pause=False, plt_show = False):
    if figure is None or axes is None or lines is None:
        return

    WINDOW_WIDTH = 0 if len(true_states[:, 0]) < 10000 else -10000
    i = 0

    i_1 = -100
    i_2 = -1

    times = times[i_1:i_2]

    # axes[0, 0].plot(times, est_states[:,0], label = "x dir_est", color = "green")
    # axes[0, 0].plot(times[WINDOW_WIDTH:], input_goal[WINDOW_WIDTH:, 0], label="x goal")
    # axes[0, 0].plot(times[WINDOW_WIDTH:], true_states[WINDOW_WIDTH:, 0], label="x dir")
    lines[i].set_data(times, input_goal[i_1:i_2, 0])
    i += 1
    lines[i].set_data(times, true_states[i_1:i_2, 0])
    i += 1
    lines[i].set_data(times, est_states[i_1:i_2, 0])
    i += 1
    # lines[i].set_text('line %d, stage %d'%(1,2))
    # i += 1

    # axes[1, 0].plot(times, est_states[:,1], label = "y dir_est" , color = "green")
    # axes[1, 0].plot(times[WINDOW_WIDTH:], input_goal[WINDOW_WIDTH:, 1], label="y goal")
    # axes[1, 0].plot(times[WINDOW_WIDTH:], true_states[WINDOW_WIDTH:, 1], label="y dir")
    lines[i].set_data(times, input_goal[i_1:i_2, 1])
    i += 1
    lines[i].set_data(times, true_states[i_1:i_2, 1])
    i += 1
    lines[i].set_data(times, est_states[i_1:i_2, 1])
    i += 1

    # axes[2, 0].plot(times, est_states[:,2], label = "z dir_est", color = "green")
    # axes[2, 0].plot(times[WINDOW_WIDTH:], input_goal[WINDOW_WIDTH:, 2], label="z goal")
    # axes[2, 0].plot(times[WINDOW_WIDTH:], true_states[WINDOW_WIDTH:, 2], label="z (altitude)")
    lines[i].set_data(times, input_goal[i_1:i_2, 2])
    i += 1
    lines[i].set_data(times, true_states[i_1:i_2, 2])
    i += 1
    lines[i].set_data(times, est_states[i_1:i_2, 2])
    i += 1

    # axes[0, 1].plot(times, np.degrees(est_states[:,6]), label = "roll_est", color = "green")
    # axes[0, 1].plot(times[WINDOW_WIDTH:], np.degrees(true_states[WINDOW_WIDTH:, 6]), label="roll")
    lines[i].set_data(times, np.degrees(true_states[i_1:i_2, 6]))
    i += 1
    lines[i].set_data(times, np.degrees(est_states[i_1:i_2, 6]))
    i += 1

    # axes[1, 1].plot(times, np.degrees(est_states[:,7]), label = "pitch_est", color = "green")
    # axes[1, 1].plot(times[WINDOW_WIDTH:], np.degrees(true_states[WINDOW_WIDTH:, 7]), label="pitch")
    lines[i].set_data(times, np.degrees(true_states[i_1:i_2, 7]))
    i += 1
    lines[i].set_data(times, np.degrees(est_states[i_1:i_2, 7]))
    i += 1

    # axes[2, 1].plot(times, np.degrees(est_states[:,8]), label = "yaw_est", color = "green")
    # axes[2, 1].plot(times[WINDOW_WIDTH:], np.degrees(yaw_goal[WINDOW_WIDTH:]), label="yaw goal")
    # axes[2, 1].plot(times[WINDOW_WIDTH:], np.degrees(true_states[WINDOW_WIDTH:, 8]), label="yaw")
    lines[i].set_data(times, overshoots[i_1:i_2,])
    i += 1

    try:
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                axes[i][j].relim()
                axes[i][j].autoscale_view()
        figure.canvas.draw()
        figure.canvas.flush_events()
    except (_tkinter.TclError):
        figure = None

   
def plot_all_results(times, true_states, est_states, torques, speeds, accels, input_goal, yaw_goal, plt_pause=False, plt_show = False):
    
    if (plt_show == True):
        i_1 = 0
        i_2 = -1
        fig1, ax1 = plt.subplots(3,2,figsize=(10,  7))
        fig1.suptitle('x, y, z, roll, pitch, yaw', fontsize=16)
        ax1[0,0].plot(times, est_states[:,0], label = "x dir_est", color = "green")
        ax1[0,0].plot(times[i_1:i_2], input_goal[i_1:i_2,0], label = "x goal")
        ax1[0,0].plot(times[i_1:i_2], true_states[i_1:i_2,0], label = "x dir")
        ax1[0,0].set_xlabel('time (s)')
        ax1[0,0].set_ylabel('x (m)')
        ax1[0,0].legend()

        ax1[1,0].plot(times, est_states[:,1], label = "y dir_est" , color = "green")
        ax1[1,0].plot(times[i_1:i_2], input_goal[i_1:i_2,1], label = "y goal")
        ax1[1,0].plot(times[i_1:i_2], true_states[i_1:i_2,1], label = "y dir")
        ax1[1,0].set_xlabel('time (s)')
        ax1[1,0].set_ylabel('y (m)')
        ax1[1,0].legend()

        # ax1[2,0].plot(input_goal[0:-1:1000,0], input_goal[0:-1:1000,1], label = "x-y goal")
        # ax1[2,0].plot(true_states[0:-1:1000,0], true_states[0:-1:1000,1], label = "true x-y")
        # ax1[2,0].set_xlabel('x (m)')
        # ax1[2,0].set_ylabel('y (m)')
        # ax1[2,0].set_ylim([0,11])
        # ax1[2,0].legend()

        ax1[2,0].plot(times, est_states[:,2], label = "z dir_est", color = "green")
        ax1[2,0].plot(times[i_1:i_2], input_goal[i_1:i_2,2], label = "z goal")
        ax1[2,0].plot(times[i_1:i_2], true_states[i_1:i_2,2], label = "z (altitude)")
        ax1[2,0].set_xlabel('time (s)')
        ax1[2,0].set_ylabel('z (m)')
        # ax1[2,0].set_ylim([0,11])
        ax1[2,0].legend()

        ax1[0,1].plot(times, np.degrees(est_states[:,6]), label = "roll_est", color = "green")
        ax1[0,1].plot(times[i_1:i_2], np.degrees(true_states[i_1:i_2,6]), label = "roll")
        ax1[0,1].set_xlabel('time (s)')
        ax1[0,1].set_ylabel('roll (deg)')
        ax1[0,1].legend()

        ax1[1,1].plot(times, np.degrees(est_states[:,7]), label = "pitch_est", color = "green")
        ax1[1,1].plot(times[i_1:i_2], np.degrees(true_states[i_1:i_2,7]), label = "pitch")
        ax1[1,1].set_xlabel('time (s)')
        ax1[1,1].set_ylabel('pitch (deg)')
        ax1[1,1].legend()

        ax1[2,1].plot(times, np.degrees(est_states[:,8]), label = "yaw_est", color = "green")
        ax1[2,1].plot(times[i_1:i_2], np.degrees(yaw_goal[i_1:i_2]), label = "yaw goal")
        ax1[2,1].plot(times[i_1:i_2], np.degrees(true_states[i_1:i_2,8]), label = "yaw")
        ax1[2,1].set_xlabel('time (s)')
        ax1[2,1].set_ylabel('yaw (deg)')
        ax1[2,1].legend()
        plt.show() 

    # fig2, ax2 = plt.subplots(3,2,figsize=(10,  7))
    # fig2.suptitle('v_x, v_y, v_z, roll_rate, pitch_rate, yaw_rate', fontsize=16)
    # ax2[0,0].plot(times, true_states[:,3], label = "x_vel")
    # ax2[0,0].plot(times, est_states[:,3], label = "x_vel_est", color = "green")
    # ax2[0,0].set_xlabel('time (s)')
    # ax2[0,0].set_ylabel('v_x (m/s)')
    # ax2[0,0].legend()

    # ax2[1,0].plot(times, true_states[:,4], label = "y_vel")
    # ax2[1,0].plot(times, est_states[:,4], label = "y_vel_est", color = "green")
    # ax2[1,0].set_xlabel('time (s)')
    # ax2[1,0].set_ylabel('v_y (m/s)')
    # ax2[1,0].legend()

    # ax2[2,0].plot(times, true_states[:,5], label = "z_vel")
    # ax2[2,0].plot(times, est_states[:,5], label = "z_vel_est", color = "green")
    # ax2[2,0].set_xlabel('time (s)')
    # ax2[2,0].set_ylabel('v_z (m/s)')
    # ax2[2,0].legend()

    # ax2[0,1].plot(times, true_states[:,9], label = "roll_rate")
    # ax2[0,1].plot(times, est_states[:,9], label = "roll_rate_est", color = "green")
    # ax2[0,1].set_xlabel('time (s)')
    # ax2[0,1].set_ylabel('phi_rate (rad/s)')
    # ax2[0,1].legend()

    # ax2[1,1].plot(times, true_states[:,10], label = "pitch_rate")
    # ax2[1,1].plot(times, est_states[:,10], label = "pitch_rate_est", color = "green")
    # ax2[1,1].set_xlabel('time (s)')
    # ax2[1,1].set_ylabel('theta_rate (rad/s)')
    # ax2[1,1].legend()

    # ax2[2,1].plot(times, true_states[:,11], label = "yaw_rate_vel")
    # ax2[2,1].plot(times, est_states[:,11], label = "yaw_rate_est", color = "green")
    # ax2[2,1].set_xlabel('time (s)')
    # ax2[2,1].set_ylabel('gamma_rate (rad/s)')
    # ax2[2,1].legend()

    # fig3, ax3 = plt.subplots(4,1, figsize=(10,  7))
    # fig3.suptitle('Torques, roll, pitch, yaw', fontsize=16)
    # ax3[0].plot(times, torques[:,0], label = "roll torque")
    # ax3[0].set_xlabel('time (s)')
    # ax3[0].set_ylabel('roll (N.m)')
    # ax3[0].legend()
    # ax3[1].plot(times, torques[:,1], label = "pitch torque")
    # ax3[1].set_xlabel('time (s)')
    # ax3[1].set_ylabel('pitch (N.m)')
    # ax3[1].legend()
    # ax3[2].plot(times, torques[:,2], label = "yaw torque")
    # ax3[2].set_xlabel('time (s)')
    # ax3[2].set_ylabel('yaw (N.m)')
    # ax3[2].legend()
    # ax3[3].plot(times, torques[:,3], label = "Vertical Thrust")
    # ax3[3].set_xlabel('time (s)')
    # ax3[3].set_ylabel('T (N)')
    # ax3[3].legend()

    # fig4, ax4 = plt.subplots(6,1, figsize=(10,  7))
    # fig4.suptitle('motor speeds', fontsize=16)
    # for idx in range(0,6):
    #     ax4[idx].plot(times, speeds[:,idx], label = "m{idx}")
    #     ax4[idx].set_xlabel('time (s)')
    #     ax4[idx].set_ylabel('m{idx} (rad/s)')
    #     ax4[idx].legend()

    # fig5, ax5 = plt.subplots(3,1, figsize=(10,  7))
    # fig5.suptitle('Accelerations, a_x, a_y, a_z', fontsize=16)
    # ax5[0].plot(times, accels[:,0], label = "a_x")
    # ax5[0].set_xlabel('time (s)')
    # ax5[0].set_ylabel('a_x (m/s^2)')
    # ax5[0].legend()
    # ax5[1].plot(times, accels[:,1], label = "a_y")
    # ax5[1].set_xlabel('time (s)')
    # ax5[1].set_ylabel('a_y (m/s^)')
    # ax5[1].legend()
    # ax5[2].plot(times, accels[:,2], label = "a_z")
    # ax5[2].set_xlabel('time (s)')
    # ax5[2].set_ylabel('a_z (m/s^2)')
    # ax5[2].legend()
    
