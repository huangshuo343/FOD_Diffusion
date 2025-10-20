# %%
import os
import json

data_base_dir = "/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/fod/UKBiobank2020lowdistortion"

# read txt
with open("/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/drc/script/hclv_data/UKBiobank_data/distortion_painting/control_data.txt", 'r') as f:
    control_data = f.readlines()
    control_data = [x.strip() for x in control_data]
    
json_list = []

for sub_id in control_data[0 : 90]:
    sub_path = os.path.join(data_base_dir, sub_id, "FOD_data_save")

    mask_path = os.path.join(sub_path, "mask.nii.gz")
    FOD_pos_nodistortion_path = os.path.join(sub_path, "FOD_pos_nodistortion.nii.gz")

    for i in range(45):
        attn_volume_path = os.path.join(sub_path, "single_volume", f"FOD_pos_nodistortion_volume{i}.nii.gz")
        target_volume_path = os.path.join(sub_path, "single_volume", f"FOD_pos_volume{i}.nii.gz")

        volume_enc_txt_path = os.path.join(sub_path, "single_volume", f"vomule_{i}.txt")

        json_list.append({"source": FOD_pos_nodistortion_path, "attn": attn_volume_path, "target": target_volume_path, "mask": mask_path, "meta": volume_enc_txt_path})



target_data_json = "/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/distortion_painting/data/diffusion_brainstem_painting_4D_training.json"

with open(target_data_json, 'w') as f:
    json.dump(json_list, f)

json_list = []

for sub_id in control_data[90 : 100]:
    sub_path = os.path.join(data_base_dir, sub_id, "FOD_data_save")

    mask_path = os.path.join(sub_path, "mask.nii.gz")
    FOD_pos_nodistortion_path = os.path.join(sub_path, "FOD_pos_nodistortion.nii.gz")

    for i in range(45):
        attn_volume_path = os.path.join(sub_path, "single_volume", f"FOD_pos_nodistortion_volume{i}.nii.gz")
        target_volume_path = os.path.join(sub_path, "single_volume", f"FOD_pos_volume{i}.nii.gz")

        volume_enc_txt_path = os.path.join(sub_path, "single_volume", f"vomule_{i}.txt")

        json_list.append({"source": FOD_pos_nodistortion_path, "attn": attn_volume_path, "target": target_volume_path, "mask": mask_path, "meta": volume_enc_txt_path})



target_data_json = "/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/distortion_painting/data/diffusion_brainstem_painting_4D_validation.json"

with open(target_data_json, 'w') as f:
    json.dump(json_list, f)

json_list = []

for sub_id in control_data[100 : ]:
    sub_path = os.path.join(data_base_dir, sub_id, "FOD_data_save")

    mask_path = os.path.join(sub_path, "mask.nii.gz")
    FOD_pos_nodistortion_path = os.path.join(sub_path, "FOD_pos_nodistortion.nii.gz")

    for i in range(45):
        attn_volume_path = os.path.join(sub_path, "single_volume", f"FOD_pos_nodistortion_volume{i}.nii.gz")
        target_volume_path = os.path.join(sub_path, "single_volume", f"FOD_pos_volume{i}.nii.gz")

        volume_enc_txt_path = os.path.join(sub_path, "single_volume", f"vomule_{i}.txt")

        json_list.append({"source": FOD_pos_nodistortion_path, "attn": attn_volume_path, "target": target_volume_path, "mask": mask_path, "meta": volume_enc_txt_path})



target_data_json = "/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/distortion_painting/data/diffusion_brainstem_painting_4D_test.json"

with open(target_data_json, 'w') as f:
    json.dump(json_list, f)

# for sub_id in control_data[100 : 101]:
#     sub_path = os.path.join(data_base_dir, sub_id, "FOD_data_save")

#     mask_path = os.path.join(sub_path, "mask.nii.gz")
#     FOD_pos_nodistortion_path = os.path.join(sub_path, "FOD_pos_nodistortion.nii.gz")

#     for i in range(45):
#         attn_volume_path = os.path.join(sub_path, "single_volume", f"FOD_pos_nodistortion_volume{i}.nii.gz")
#         target_volume_path = os.path.join(sub_path, "single_volume", f"FOD_pos_volume{i}.nii.gz")

#         volume_enc_txt_path = os.path.join(sub_path, "single_volume", f"vomule_{i}.txt")

#         json_list.append({"source": FOD_pos_nodistortion_path, "attn": attn_volume_path, "target": target_volume_path, "mask": mask_path, "meta": volume_enc_txt_path})



# target_data_json = "/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/distortion_painting/data/diffusion_brainstem_painting_4D_test1data.json"

# with open(target_data_json, 'w') as f:
#     json.dump(json_list, f)