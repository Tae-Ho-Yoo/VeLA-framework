# Realtime test codes

Framework server file name:
- deploy_realtime.py

Run order:

Terminal 1:
python deploy_realtime.py

Terminal 2:
python task_monitor_qr_slots.py --camera 0 --config task_monitor_slots_config.json

Terminal 3:
python mock_experiment_controller.py --sequence current_sequence.json

Purpose:
- Test webpage/random sequence -> mock controller -> task monitor -> deploy_realtime.py /update_speed.
- UR5e is not required in this stage.
