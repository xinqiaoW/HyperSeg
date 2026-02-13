import numpy as np
import torch.nn.functional as F
import os
import cv2
import torch

"""
Utilizing HyperFree for hyperspectral classification task directly without fine tuning.
We develop different Prompt—Mask—Feature interaction workflows for each task.
"""

def hyperspectral_classification(mask_generator, image, few_shots, spectral_lengths, feature_index_id, GSD = torch.tensor([1.0]), open_set_thresh = 0):
    """
    Utilizing HyperFree for hyperspectral classification directly without fine tuning.
    Args:
        mask_generator: object instance from SamAutomaticMaskGenerator
        image: input hyperspectral image with shape [H, W, C] in range [0,255]
        few_shots: a list, storing binary maps for each class, where the non-zero location represents the corresponding sample
        spectral_lengths: a list, storing wavelengths for each hyperspectral channel 
        feature_index_id: deciding which stage of encoder features to use, in range [0, 5]
        GSD: ground sampling distance (m/pixel). list, such as [1.0] or tensor, such as torch.tensor([1.0])
        open_set_thresh: distinguishing unknown classes out if open_set_thresh > 0, which represents the cosine similarity threshold at feature level

    Returns:
        classification_maps_each_class: a list storing binary classification maps for each class separately
        classification_map: grey map with shape [H, W], where each pixel id represents a object class
    """

    assert open_set_thresh >= 0 and open_set_thresh <=1
    anns, multi_stage_features = mask_generator.generate(image, spectral_lengths, GSD)
    mask = mask_generator.anns2mask(anns)
    img = mask_generator.show_anns(anns)
    cv2.imwrite('mask.png', (img * 255).astype(np.uint8))

    # mask_generator.predictor.set_image(image, True, spectral_lengths, GSD)
    # all_features = mask_generator.predictor.model.image_encoder.multi_stage_features[feature_index_id]
    all_features = multi_stage_features[feature_index_id]
    all_features = all_features.detach().cpu()
    all_features = F.interpolate(all_features, (max(mask.shape[1], mask.shape[2]),max(mask.shape[1], mask.shape[2]) ))
    all_features = all_features[:,:, :mask.shape[1], :mask.shape[2]]
     
    target_features = []
    classification_maps_each_class = []
    for few_shot_label in few_shots:
        target_feature = []

        target_locs = np.where(few_shot_label == 1)
        for loc_index in range(len(target_locs[0])):
            # mask_index = np.where(mask[:,target_locs[0][loc_index], target_locs[1][loc_index]] == 1)[0]
            # assert mask_index.size != 0, ("The setting hyper-parameters lead to no mask in given target location")

            # few_shot_label = mask[mask_index.tolist(), :, :][0,:,:]
            # target_locs_t = np.where(few_shot_label == 1)
        
            # target_feature_t = all_features[0:1, :, (target_locs_t[0]), (target_locs_t[1])]
            target_feature_t = all_features[0, :, target_locs[0][loc_index], target_locs[1][loc_index]]
            # target_feature_t = target_feature_t.mean((2))[0,:].detach().cpu().numpy()

            # Another way to compute target_feature_t
            # target_feature_t = all_features[0, :, target_locs[0][loc_index], target_locs[1][loc_index]]
            target_feature.append(target_feature_t)
        
        target_features.append(target_feature)
        classification_maps_each_class.append(np.zeros((mask.shape[1], mask.shape[2])))

    mask_number = mask.shape[0]
    classification_map = np.zeros((mask.shape[1], mask.shape[2]), dtype=np.uint8)
    for i in range(mask_number):
        seg_mask = mask[-2-i:-1-i,:,:]
        locs = np.where(seg_mask == 1)
        seg_mask_feature = all_features[:,:, locs[1], locs[2]].mean(2)[0,:]

        best_index = -1
        highest_score = -1
        for j in range(len(target_features)):
            target_feature = target_features[j]
            for target_feature_t in target_feature:
                cosine = mask_generator.cosine_similarity(seg_mask_feature.detach().cpu().numpy(), target_feature_t)
                if cosine > highest_score:
                    highest_score = cosine
                    best_index = j

        if cosine > open_set_thresh:
            classification_maps_each_class[best_index][locs[1], locs[2]] = 1
            classification_map[locs[1], locs[2]] = best_index
        else:
            classification_maps_each_class[best_index][locs[1], locs[2]] = 0
        
    return classification_maps_each_class, classification_map


def show_anns(anns, save_dir=''):
    if len(anns) == 0:
        print("len=0")
        return
    class_id = 1
    for mask in anns:
        res = np.zeros((mask.shape[0], mask.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        locs = np.where(mask[:,:] == True)
        res[locs[0], locs[1], :] = np.array(color_mask)*255
        save_path = os.path.join(save_dir, str(class_id) + '.png')
        class_id += 1
        cv2.imwrite(save_path, res)

# def save_anns(anns, save_dir='',name = 'result'):
#     if len(anns) == 0:
#         return
#     save_label_path = os.path.join(save_dir, name + '.tif')
#     write_img(anns, save_label_path)

def enhance_contrast_histogram(hsi):
    hsi = np.array(hsi, dtype=np.float32)    
    H, W, C = hsi.shape    
    enhanced_hsi = np.zeros_like(hsi, dtype=np.float32)
    
    for c in range(C):
        band = hsi[:, :, c]   
        band_scaled = cv2.normalize(band, None, 0, 500, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        enhanced_band = cv2.equalizeHist(band_scaled)
        enhanced_hsi[:, :, c] = enhanced_band
    
    return enhanced_hsi
