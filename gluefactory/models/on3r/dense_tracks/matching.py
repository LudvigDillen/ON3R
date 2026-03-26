import torch


def get_match_data(scaler_query, scalers_ref, roma_model, img_paths, device, num=3000):
    tuple_length = scalers_ref.shape[0]
    height_q = scaler_query[1]
    width_q = scaler_query[0]
    query_kpts_train = []
    ref_kpts_train_ = []
    all_certainties = torch.full((tuple_length, num), -1.0, dtype=torch.float32, device=device)
    all_query_kpts = torch.full((tuple_length, num, 2), -1.0, dtype=torch.float32, device=device)
    all_ref_kpts = torch.full((tuple_length, num, 2), -1.0, dtype=torch.float32, device=device)
    out_res = roma_model.get_output_resolution()  # tuple with two items (w, h)
    all_warps = torch.empty((tuple_length, out_res[0], out_res[1]*2, 4), dtype=torch.float32, device=device)
    all_matches = torch.full((tuple_length, num, 4), -2.0, dtype=torch.float32, device=device)
    for j in range(tuple_length):
        warp, certainty = roma_model.match(img_paths[0], img_paths[j + 1], device=device)
        all_warps[j] = warp  # Store the warp for later use

        matches, certainty = roma_model.sample(warp, certainty, num=num)
        n_matches = len(matches)
        all_matches[j, :n_matches] = matches
        all_certainties[j, :n_matches] = certainty

        # Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
        height_r = scalers_ref[j, 1]
        width_r = scalers_ref[j, 0]

        kpts_query, kpts_ref = roma_model.to_pixel_coordinates(matches, height_q, width_q, height_r, width_r)
        all_query_kpts[j, :n_matches] = kpts_query
        all_ref_kpts[j, :n_matches] = kpts_ref
        query_kpts_train.append(kpts_query)
        ref_kpts_train_.append(kpts_ref)
    return (all_warps, all_matches, all_certainties, all_query_kpts, all_ref_kpts)
