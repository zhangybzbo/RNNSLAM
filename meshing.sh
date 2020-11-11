# name="/media/zhangyb/Seagate_Backup_Plus_Drive/frames/"
name="/home/zhangyb/projects/ColonData/sequences/"
video=""
#video="Auto_A_Feb05_09-17-13"
num="020"
# "$name$video.mov/$num/reproduce/tum_refined_145000/" trajectory.txt \
# --export_mesh "./outputs/mesh_$video.$num.obj" \
./build/applications/surfel_meshing/SurfelMeshing \
	"$name/$num/reproduce/tum_refined_145000/" trajectory.txt \
	--follow_input_camera false --depth_valid_region_radius 160 \
	--export_mesh "./outputs/newRNN/mesh_$num.obj" \
	--outlier_filtering_frame_count 2 \
	--outlier_filtering_required_inliers 1 \
	--observation_angle_threshold_deg 90 \
	--sensor_noise_factor 0.3 \
	--hide_camera_frustum --render_window_default_width 640 --render_window_default_height 640 \
	--max_depth 2.5 \
	--bilateral_filter_radius_factor 5
