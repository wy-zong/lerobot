lerobot-teleoperate --robot.type=so101_follower --robot.port=COM4 --robot.id=so101_follower_left --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=so101_leader_left


lerobot-teleoperate --robot.type=so101_follower --robot.port=COM6 --robot.id=so101_follower_right --teleop.type=so101_leader --teleop.port=COM7 --teleop.id=so101_leader_right

lerobot-teleoperate `
   --robot.type=bi_so100_follower `
   --robot.left_arm_port=COM5 `
   --robot.right_arm_port=COM6 `
   --robot.id=bimanual_follower `
   --teleop.type=bi_so100_leader `
   --teleop.left_arm_port=COM4 `
   --teleop.right_arm_port=COM7 `
   --teleop.id=bimanual_leader





lerobot-record --robot.type=so101_follower --robot.port=COM6 --robot.id=so101_follower_arm --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera3: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" --display_data=true --dataset.repo_id="local/eval_Fold_the_rag0209TEST1" --dataset.single_task="Fold the rag" --dataset.num_episodes=50 --dataset.episode_time_s=100 --dataset.reset_time_s=5 --dataset.push_to_hub=false --policy.path="C:\Users\ccu\mujoco_ur5_graph\outputs\model\Fold-the-rag-transforms-only-expert=true\checkpoints\050000\pretrained_model" --policy.device=cuda --teleop.type=so101_leader --teleop.port=COM7 --teleop.id=so101_leader_arm


lerobot-record `
     --robot.type=bi_so100_follower `
     --robot.left_arm_port=COM5 `
     --robot.right_arm_port=COM6 `
     --robot.id=bimanual_follower `
     --robot.cameras='{"camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "camera3": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' `
     --teleop.type=bi_so100_leader `
     --teleop.left_arm_port=COM4 `
     --teleop.right_arm_port=COM7 `
     --teleop.id=bimanual_leader `
     --display_data=true `
     --dataset.repo_id=local/full-fold-the-rag-newv-lerobot `
     --dataset.root=C:\Users\ccu\mujoco_ur5_graph\outputs\dataset\full-fold-the-rag-newv-lerobot0301 `
     --dataset.num_episodes=50 `
     --dataset.single_task="full fold the rag" `
     --dataset.video=true `
     --dataset.push_to_hub=false `
     --dataset.episode_time_s=180 `
     --dataset.reset_time_s=5


lerobot-record `
     --robot.type=bi_so100_follower `
     --robot.left_arm_port=COM5 `
     --robot.right_arm_port=COM6 `
     --robot.id=bimanual_follower `
     --robot.cameras='{"camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "camera3": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' `
     --display_data=true `
     --dataset.repo_id=local/eval_Fold_the_rag0302TEST1 `
     --dataset.single_task="full Fold the rag" `
     --dataset.num_episodes=50 `
     --dataset.episode_time_s=300 `
     --dataset.reset_time_s=5 `
     --dataset.push_to_hub=false `
     --policy.path=C:/Users/ccu/mujoco_ur5_graph/outputs/model/checkpoints/020000/pretrained_model `
     --policy.device=cuda `
     --teleop.type=bi_so100_leader `
     --teleop.left_arm_port=COM4 `
     --teleop.right_arm_port=COM7 `
     --teleop.id=bimanual_leader


  $env:HF_LEROBOT_HOME='C:\Users\ccu\mujoco_ur5_graph\outputs'
  python -m lerobot.scripts.lerobot_edit_dataset `
    --repo_id "full-fold-the-rag-parquet-merged" `
    --operation.type merge `
    --operation.repo_ids "['full-fold-the-rag-parquet','full-fold-the-rag-parquet-c']"

lerobot-train   --dataset.root="/home/wy/outputs/full-fold-the-rag-jpeg-parquet"   --dataset.repo_id="local/full-Fold-the-rag-jpeg-parquet"   --policy.path="lerobot/smolvla_base"   --policy.device=cuda  --steps=50000 --num_workers=4   --output_dir="/home/wy/outputs/train/full-Fold-the-rag-jpeg-pa
rauet-vlm2.2b"   --job_name="full-Fold-the-rag"   --policy.repo_id="local/full-fold-the-rag-jpeg-parquet"   --policy.push_to_hub=false   --wandb.enable=false --dataset.imag
e_transforms.enable=true 

lerobot-train   --dataset.root="/home/wy/outputs/full-fold-the-rag-jpeg-parquet"   --dataset.repo_id="local/full-Fold-the-rag-jpeg-parquet"   --policy.path="lerobot/smolvla_base"   --policy.device=cuda  --steps=50000 --num_workers=4   --output_dir="/home/wy/outputs/train/full-Fold-the-rag-jpeg-pa
rauet-vlm2.2b"   --job_name="full-Fold-the-rag"   --policy.repo_id="local/full-fold-the-rag-jpeg-parquet"   --policy.push_to_hub=false   --wandb.enable=false --dataset.imag
e_transforms.enable=true --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-2.2B

$env:HF_LEROBOT_HOME='C:\Users\ccu\mujoco_ur5_graph\outputs'
python -m lerobot.scripts.lerobot_edit_dataset `
    --repo_id "full-fold-the-rag-parquet-merged0222" `
    --operation.type merge `
    --operation.repo_ids "['dataset/full-fold-the-rag-jpeg-parquet-0222','full-fold-the-rag-jpeg-parquet']"




python -m lerobot.scripts.lerobot_record_hi `
        --robot.type=so101_follower `
        --robot.port=COM5 `
        --robot.id=so101_follower_arm `
        --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera3: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" `
        --display_data=true `
        --dataset.repo_id="local/eval_fold_the_rag0226TEST1" `
        --dataset.single_task="Fold the rag" `
        --dataset.num_episodes=5 `
        --dataset.episode_time_s=1000 `
        --dataset.reset_time_s=5 `
        --dataset.push_to_hub=false `
        --policy.path="C:\Users\ccu\mujoco_ur5_graph\outputs\model\Fold-the-rag-transforms-only-expert=true\checkpoints\050000\pretrained_model" `
        --policy.device=cuda `
        --teleop.type=so101_leader `
        --teleop.port=COM4 `
        --teleop.id=so101_leader_arm `
        --human_intervention=true `
        --dataset.video=false

python -m lerobot.scripts.lerobot_record_hi `
     --robot.type=bi_so100_follower `
     --robot.left_arm_port=COM5 `
     --robot.right_arm_port=COM6 `
     --robot.id=bimanual_follower `
     --robot.cameras='{"camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "camera3": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' `
     --display_data=true `
     --dataset.repo_id=local/eval_Fold_the_rag0226TEST1 `
     --dataset.single_task="full Fold the rag" `
     --dataset.num_episodes=50 `
     --dataset.episode_time_s=300 `
     --dataset.reset_time_s=5 `
     --dataset.push_to_hub=false `
     --policy.path=C:\Users\ccu\mujoco_ur5_graph\outputs\model\040000\pretrained_model `
     --policy.device=cuda `
     --teleop.type=bi_so100_leader `
     --teleop.left_arm_port=COM4 `
     --teleop.right_arm_port=COM7 `
     --teleop.id=bimanual_leader `
     --human_intervention=true `
     --dataset.video=false


lerobot-train `
   --dataset.root="C:\Users\ccu\mujoco_ur5_graph\outputs\full-fold-the-rag-parquet-merged0222" `
   --dataset.repo_id=local/full-fold-the-rag-parquet-merged0222 `
   --policy.type=sarm `
   --policy.annotation_mode=single_stage `
   --policy.image_key=observation.images.base `
   --output_dir=C:\Users\ccu\mujoco_ur5_graph\outputs/train/sarm_single `
   --batch_size=32 `
   --steps=5000 `
   --wandb.enable=true `
   --wandb.project=sarm `
   --policy.repo_id=wy_SARM/wy_SARM

lerobot-edit-dataset `
     --repo_id C:/Users/ccu/mujoco_ur5_graph/outputs/full-fold-the-rag-parquet-merged0222 `
     --operation.type convert_image_to_video `
     --operation.output_dir D:/full-fold-the-rag-parquet-merged0222-video `
     --operation.vcodec=auto                 

#0301最新版本的lerobot影片編碼幾乎可以瞬間完成
lerobot-record `
      --robot.type=so101_follower `
      --robot.port=COM5 `
      --robot.id=so101_follower_arm `
      --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera3: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" `
      --display_data=true `
      --dataset.repo_id="local/fold_the_rag0301TEST1" `
      --dataset.root="C:\Users\ccu\mujoco_ur5_graph\outputs\full-fold-the-rag0301" `
      --dataset.single_task="Fold the rag" `
      --dataset.num_episodes=5 `
      --dataset.episode_time_s=1000 `
      --dataset.reset_time_s=5 `
      --dataset.push_to_hub=false `
      --teleop.type=so101_leader `
      --teleop.port=COM4 `
      --teleop.id=so101_leader_arm `
      --dataset.video=true `
      --dataset.vcodec=auto `
      --dataset.streaming_encoding=true `
      --dataset.encoder_threads=2

lerobot-record `
     --robot.type=bi_so_follower `
     --robot.left_arm_config.port=COM5 `
     --robot.right_arm_config.port=COM8 `
     --robot.id=bimanual_follower `
     --teleop.type=bi_so_leader `
     --teleop.left_arm_config.port=COM4 `
     --teleop.right_arm_config.port=COM9 `
     --teleop.id=bimanual_leader `
     --robot.cameras='{"camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 60}, "camera2": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 60}, "camera3": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 60}}' `
     --display_data=true `
     --dataset.repo_id=local/full-fold-the-rag-newv-lerobot `
     --dataset.root=C:\Users\ccu\mujoco_ur5_graph\outputs\dataset\bi_so101_fold-the-rag-newv-lerobot0323 `
     --dataset.num_episodes=50 `
     --dataset.single_task="full fold the rag" `
     --dataset.video=true `
     --dataset.push_to_hub=false `
     --dataset.episode_time_s=180 `
     --dataset.reset_time_s=5 `
     --dataset.vcodec=auto `
     --dataset.streaming_encoding=true `
     --dataset.encoder_threads=2


lerobot-record `
     --robot.type=bi_so_follower `
     --robot.left_arm_config.port=COM6 `
     --robot.right_arm_config.port=COM8 `
     --robot.id=bimanual_follower `
     --teleop.type=bi_so_leader `
     --teleop.left_arm_config.port=COM4 `
     --teleop.right_arm_config.port=COM9 `
     --teleop.id=bimanual_leader `
     --robot.cameras='{"camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "camera2": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}, "camera3": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' `
     --display_data=true `
     --dataset.repo_id=wuc1/bi_so101_flatten-and-fold-the-rag-then-place-0502 `
     --dataset.num_episodes=50 `
     --dataset.single_task="flatten and fold the rag then place" `
     --dataset.video=true `
     --dataset.push_to_hub=true `
     --dataset.episode_time_s=300 `
     --dataset.reset_time_s=5 `
     --dataset.vcodec=auto `
     --dataset.streaming_encoding=true `
     --dataset.encoder_threads=2 `
          --robot.cameras='{"camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "camera3": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}, "camera2": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}' `
     


python -m lerobot.async_inference.robot_client `
    --robot.type=bi_so_follower `
    --robot.left_arm_config.port=COM6 `
    --robot.right_arm_config.port=COM8 `
    --robot.id=bimanual_follower `
    --robot.cameras='{"left_camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "right_camera2": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}, "left_camera3": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' `
    --task="flatten and fold the rag then place" `
    --server_address=192.168.50.198:8080 `
    --policy_type=pi05 `
    --pretrained_name_or_path=wuc1/bi_so101_flatten-and-fold-the-rag-then-place-pi05-no-image_transforms-2 `
    --policy_device=cuda `
    --client_device=cuda `
    --actions_per_chunk=30 `
    --chunk_size_threshold=0.0 `
    --aggregate_fn_name=latest_only `
    --debug_visualize_queue_size=true `
    --rename_map='{"observation.images.left_camera1": "observation.images.camera1", "observation.images.left_camera3": "observation.images.camera3", "observation.images.right_camera2": "observation.images.camera2"}' `
    --fps=30 `
    

lerobot-record `
       --robot.type=bi_so_follower `
       --robot.left_arm_config.port=COM5 `
       --robot.right_arm_config.port=COM8 `
       --robot.id=bimanual_follower `
       --teleop.type=bi_so_leader `
       --teleop.left_arm_config.port=COM4 `
       --teleop.right_arm_config.port=COM9 `
       --teleop.id=bimanual_leader `
       --robot.left_arm_config.cameras='{"camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "camera3": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' `
       --robot.right_arm_config.cameras='{"camera2": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}' `
       --display_data=true `
       --dataset.repo_id=local/eval_bi_so101_flatten-and-fold-the-rag-then-place-0416-0417-merge0420-model-smolvla_0422EST1 `
       --dataset.num_episodes=5 `
       --dataset.single_task="flatten and fold the rag then place" `
       --dataset.video=true `
       --dataset.push_to_hub=false `
       --dataset.episode_time_s=300 `
       --dataset.reset_time_s=5 `
       --dataset.vcodec=auto `
       --dataset.streaming_encoding=true `
       --dataset.encoder_threads=2 `
       --policy.path=wuc1/bi_so101_flatten-and-fold-the-rag-then-place-0416-0417-merge0420-model `
       --policy.device=cuda `
       --dataset.rename_map='{"observation.images.left_camera1": "observation.images.camera1", "observation.images.left_camera3": "observation.images.camera3", "observation.images.right_camera2": "observation.images.camera2"}'


python C:/Users/ccu/mujoco_ur5_graph/lerobot/examples/hil/hil_data_collection.py `
    --robot.type=bi_so_follower `
    --robot.left_arm_config.port=COM5 `
    --robot.right_arm_config.port=COM8 `
    --robot.id=bimanual_follower `
    --teleop.type=bi_so_leader `
    --teleop.left_arm_config.port=COM4 `
    --teleop.right_arm_config.port=COM9 `
    --teleop.id=bimanual_leader `
    --robot.left_arm_config.cameras='{"camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "camera3": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' `
    --robot.right_arm_config.cameras='{"camera2": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}' `
    --policy.path="C:/Users/ccu/mujoco_ur5_graph/outputs/model/03-53-32_full-fold-the-rag-RABC/checkpoints/200000/pretrained_model" `
    --dataset.repo_id=local/hil-rtc-dataset-0415 `
    --dataset.single_task="flatten and fold the rag" `
    --dataset.fps=30 `
    --dataset.episode_time_s=1000 `
    --dataset.num_episodes=50 `
    --dataset.rename_map='{"observation.images.left_camera1": "observation.images.camera1", "observation.images.left_camera3": "observation.images.camera3", "observation.images.right_camera2": "observation.images.camera2"}' `
    --interpolation_multiplier=3 `
    --dataset.push_to_hub=false `
    --rtc.enabled=true `
    --rtc.execution_horizon=20 `
    --rtc.max_guidance_weight=5.0 `
    --rtc.prefix_attention_schedule=LINEAR `

lerobot-rollout `
       --strategy.type=dagger `
       --strategy.record_autonomous=true `
       --strategy.num_episodes=5 `
       --strategy.model_test_mode=true `
       --robot.type=bi_so_follower `
       --robot.left_arm_config.port=COM7 `
       --robot.right_arm_config.port=COM8 `
       --robot.id=bimanual_follower `
       --teleop.type=bi_so_leader `
       --teleop.left_arm_config.port=COM4 `
       --teleop.right_arm_config.port=COM9 `
       --teleop.id=bimanual_leader `
       --robot.left_arm_config.cameras='{"camera1": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}, "camera3": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}, "camera4": {"type": "opencv", "index_or_path": 0, "width": 320, "height": 240, "fps": 30}}' `
       --robot.right_arm_config.cameras='{"camera2": {"type": "opencv", "index_or_path": 3, "width": 320, "height": 240, "fps": 30}}' `
       --display_data=true `
       --dataset.repo_id=local/rollout_bi_so101_ffp_RA-BC  `
       --dataset.single_task="rotate the rag 90 degrees" `
       --dataset.video=true `
       --dataset.push_to_hub=false `
       --dataset.episode_time_s=300 `
       --dataset.reset_time_s=5 `
       --dataset.streaming_encoding=true `
       --dataset.encoder_threads=2 `
       --policy.path="wuc1/bi_so101_ffp_RA-BC" `
       --sarm_model_path="wuc1/sarm_current_only_full_dataset"

       --inference.type=rtc `
       --inference.rtc.enabled=true `
       --inference.rtc.execution_horizon=10 `
       --fps=15 `
       --use_torch_compile=true `
       
       --rename_map='{"observation.images.left_camera1": "observation.images.camera1", "observation.images.left_camera3": "observation.images.camera3", "observation.images.right_camera2": "observation.images.camera2"}'

lerobot-record `
     --robot.type=bi_so_follower `
     --robot.left_arm_config.port=COM7 `
     --robot.right_arm_config.port=COM8 `
     --robot.id=bimanual_follower `
     --teleop.type=bi_so_leader `
     --teleop.left_arm_config.port=COM4 `
     --teleop.right_arm_config.port=COM9 `
     --teleop.id=bimanual_leader `
     --robot.left_arm_config.cameras='{"camera1": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 60}, "camera3": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 60}, "camera4": {"type": "opencv", "index_or_path": 0, "width": 320, "height": 240, "fps": 60}}' `
     --robot.right_arm_config.cameras='{"camera2": {"type": "opencv", "index_or_path": 3, "width": 320, "height": 240, "fps": 60}}' `
     --display_data=true `
     --dataset.repo_id=wuc1/bi_so101_ffp_free_style  `
     --dataset.num_episodes=80 `
     --dataset.single_task="flatten and fold the rag then place" `
     --dataset.video=true `
     --dataset.push_to_hub=true `
     --dataset.episode_time_s=300 `
     --dataset.reset_time_s=5 `
     --dataset.streaming_encoding=true `
     --dataset.encoder_threads=2 `
     --dataset.fps=60 `
     --dataset.push_to_hub=true


# Delete episodes and save to a new dataset (preserves original)
lerobot-edit-dataset `
     --repo_id wuc1/bi_so101_flatten-and-fold-the-rag-then-place-0416-0417-merge `
    --new_repo_id wuc1/bi_so101_flatten-and-fold-the-rag-then-place-0416-0417-merge-115s_filtered `
    --operation.type delete_episodes `
    --operation.episode_indices "[0, 3, 9, 13, 15, 16, 17, 19, 21, 22, 23, 24, 25, 26, 27, 29, 30, 33, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 48, 52, 53, 54, 55, 56, 58]"


lerobot-edit-dataset `
    --new_repo_id wuc1/bi_so101_ffp_60fps_merge `
    --operation.type merge `
    --push_to_hub=true `
    --operation.repo_ids "['wuc1/bi_so101_flatten-and-fold-the-rag-then-place_20260530_214034', 'wuc1/bi_so101_ffp-20260523_181127', 'wuc1/bi_so101_ffp-green_20260601_211437', 'wuc1/bi_so101_ffp_20260601_191307', 'wuc1/bi_so101_ffp-green_20260601_233712']"

python -m lerobot.rl.crop_dataset_roi `
    --repo-id wuc1/bi_so101_flatten-and-fold-the-rag-then-place-0416-0417-merge-115s_filtered `
    --new-repo-id wuc1/bi_so101_115s_filtered_roi `
    --video-backend pyav
 
  
lerobot-rollout `
     --strategy.type=base `
     --robot.type=bi_so_follower `
     --robot.left_arm_config.port=COM7 `
     --robot.right_arm_config.port=COM8 `
     --robot.id=bimanual_follower `
     --robot.left_arm_config.cameras='{"camera1": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}, "camera3": {"type": "opencv", "index_or_path": 2,"width": 640, "height": 480, "fps": 30}, "camera4": {"type": "opencv", "index_or_path": 0, "width": 320, "height": 240, "fps": 30}}' `
     --robot.right_arm_config.cameras='{"camera2": {"type": "opencv", "index_or_path": 3, "width": 320, "height": 240, "fps": 30}}' `
     --display_data=true `
     --policy.path="wuc1/bi_so101_ffp_4cam_60fps_tRTC" `

     --inference.type=rtc `
     --inference.type=rtc `
     --inference.rtc.enabled=true `
     --inference.rtc.execution_horizon=20 `
     --fps=15 `
     --interpolation_multiplier=2

python C:\Users\ccu\mujoco_ur5_graph\lerobot\src\lerobot\scripts\lerobot_vlm_inspect.py `
     --policy.path=wuc1/bi_so101_ffp_4cam_60fps_32L `
     --robot.type=bi_so_follower `
     --robot.left_arm_config.port=COM7 `
     --robot.right_arm_config.port=COM8 `
     --robot.id=bimanual_follower `
     --robot.left_arm_config.cameras='{"camera1": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}, "camera3": {"type": "opencv", "index_or_path": 2,"width": 640, "height": 480, "fps": 30}, "camera4": {"type": "opencv", "index_or_path": 0, "width": 320, "height": 240, "fps": 30}}' `
     --robot.right_arm_config.cameras='{"camera2": {"type": "opencv", "index_or_path": 3, "width": 320, "height": 240, "fps": 30}}' `
     --mode=answer `
     --fps=10 `
     --device=cuda `
     --display_data=true


