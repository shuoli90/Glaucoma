import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import GroupShuffleSplit


def split_groups_into_train_test(dataframe, group_column, test_size=0.2, random_state=None):
    group_shuffle_split = GroupShuffleSplit(test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(group_shuffle_split.split(dataframe, groups=dataframe[group_column]))

    train_df = dataframe.iloc[train_indices]
    test_df = dataframe.iloc[test_indices]
    return train_df, test_df


def compute(df, random_seed):
    # Split the groups in the DataFrame into training and testing sets
    train_df, test_df = split_groups_into_train_test(df, 'visit_ids', test_size=0.5, random_state=random_seed)

    cal_NCM = 1 - train_df['pred'].to_numpy()
    cal_label = train_df['label'].to_numpy()
    cal_NCM[cal_label == 0] = 1 - cal_NCM[cal_label == 0]
    test_NCM = 1 - test_df['pred'].to_numpy()
    test_label = test_df['label'].to_numpy()
    test_NCM[test_label == 0] = 1 - test_NCM[test_label == 0]

    threshold_pos = np.quantile(cal_NCM[cal_label == 1], 0.95, method='higher')
    threshold_neg = np.quantile(cal_NCM[cal_label == 0], 0.95, method='higher')
    threshold1 = np.max([threshold_pos, threshold_neg])
    threshold2 = np.quantile(cal_NCM, 0.95, method='higher')

    print('positive threshold')
    positive_coverage = np.mean(test_NCM[test_label==1] <= threshold_pos)
    print('coverage', positive_coverage)
    scores = np.stack([test_NCM[test_label==1], 1-test_NCM[test_label==1]], axis=1)
    positive_size = np.mean(np.sum(scores <= threshold_pos, axis=1))
    print('size', positive_size)

    print('negative threshold')
    negative_coverage = np.mean(test_NCM[test_label==0] <= threshold_neg)
    print('coverage', negative_coverage)
    scores = np.stack([test_NCM[test_label==0], 1-test_NCM[test_label==0]], axis=1)
    negative_size = np.mean(np.sum(scores <= threshold_neg, axis=1))
    print('size', negative_size)

    print('maximum threshold')
    maximum_coverage = np.mean(test_NCM <= threshold1)
    print('coverage', maximum_coverage)
    scores = np.stack([test_NCM, 1-test_NCM], axis=1)
    maximum_size = np.mean(np.sum(scores <= threshold1, axis=1))
    print('size', maximum_size)
    
    print('overall threshold')
    overall_coverage = np.mean(test_NCM <= threshold2)
    print('coverage', overall_coverage)
    scores = np.stack([test_NCM, 1-test_NCM], axis=1)
    overall_size = np.mean(np.sum(scores <= threshold2, axis=1))
    print('size', overall_size)
    print()
    
    # group test_df by visit_ids
    coverages_intersection = []
    coverages_union = []
    coverages_maximum = []
    sizes_intersection = []
    sizes_union = []
    sizes_maximum = []
    coverages_intra_intersection_inter_union = []
    sizes_intra_intersection_inter_union = []
    groups = test_df.groupby('visit_ids')
    # iterate over each group
    for idx, threshold in enumerate([threshold1, threshold2]):
        print(f'Threshold {idx+1}')
        for name, group in groups:
            # get the indices of the group
            indices = group.index
            # get the NCM scores of the group
            group_NCM = 1 - group['pred'].to_numpy()
            # get the labels of the group
            group_label = group['label'].to_numpy()[0]
            # one-hot encode the labels
            group_label = np.eye(2)[group_label]
            # get the maximum NCM score of the group
            max_NCM = np.max(group_NCM)
            
            # construct a N*2 matrix with the first column being the complement of NCM scores 
            # and the second column being the NCM scores
            scores = np.stack([1-group_NCM, group_NCM], axis=0).T
            includes = scores <= threshold
            # A. Take the intersection
            includes_intersection = np.all(includes, axis=0)
            coverage_intersection = np.sum(includes_intersection * group_label)
            coverages_intersection.append(coverage_intersection)
            size_intersection = np.sum(includes_intersection)
            sizes_intersection.append(size_intersection)
            # B. Take the union
            includes_union = np.any(includes, axis=0)
            coverage_union = np.sum(includes_union * group_label)
            coverages_union.append(coverage_union)
            size_union = np.sum(includes_union)
            sizes_union.append(size_union)
            # C. Take the maximum
            includes_maximum = np.max(includes, axis=0)
            coverage_maximum = np.sum(includes_maximum * group_label)
            coverages_maximum.append(coverage_maximum)
            size_maximum = np.sum(includes_maximum)
            sizes_maximum.append(size_maximum)

            # eye_groups = group.groupby('left')
            # includes_union = np.zeros(2)
            # for name, eye_group in eye_groups:
            #     eye_NCM = 1 - eye_group['pred'].to_numpy()
            #     # get the labels of the group
            #     eye_label = eye_group['label'].to_numpy()[0]
            #     # one-hot encode the labels
            #     eye_label = np.eye(2)[eye_label]
            #     # get the maximum NCM score of the group
            #     max_NCM = np.max(group_NCM)
            #     # construct a N*2 matrix with the first column being the complement of NCM scores 
            #     # and the second column being the NCM scores
            #     scores = np.stack([1-eye_NCM, eye_NCM], axis=0).T
            #     includes = scores <= threshold

            #     # A. Take the intersection
            #     includes_intersection = np.all(includes, axis=0)
            #     includes_union = np.logical_or(includes_union, includes_intersection)

            # coverage_union = np.sum(includes_union * group_label)
            # coverages_intra_intersection_inter_union.append(coverage_union)
            # size_union = np.sum(includes_union)
            # sizes_intra_intersection_inter_union.append(size_union)
        
        print('average coverage of intersection',
              np.mean(coverages_intersection))
        print('average size of intersection',
              np.mean(sizes_intersection))
        print('average coverage of union',
              np.mean(coverages_union))
        print('average size of union',
              np.mean(sizes_union))
        print('average coverage of maximum',
              np.mean(coverages_maximum))
        print('average size of maximum',
              np.mean(sizes_maximum))
        # print('average coverage of intra-intersection-inter-union',
        #       np.mean(coverages_intra_intersection_inter_union))
        # print('average size of intra-intersection-inter-union',
        #         np.mean(sizes_intra_intersection_inter_union))
        print()
        return positive_coverage, positive_size, \
               negative_coverage, negative_size, \
               maximum_coverage, maximum_size, \
               overall_coverage, overall_size, \
               np.mean(coverages_intersection), np.mean(sizes_intersection), \
               np.mean(coverages_union), np.mean(sizes_union), \
               np.mean(coverages_maximum), np.mean(sizes_maximum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='informative_images.json')
    args = parser.parse_args()

    # Read the DataFrame from the JSON file
    df = pd.read_json(args.json_file, orient='records')
    random_seeds = list(range(30))

    positve_coverages = []
    positve_sizes = []
    negative_coverages = []
    negative_sizes = []
    overall_coverages = []
    overall_sizes = []
    maximum_coverages = []
    maximum_sizes = []
    coverages_intersection = []
    sizes_intersection = []
    coverages_union = []
    sizes_union = []
    coverages_maximum = []
    sizes_maximum = []
    

    for random_seed in random_seeds:
        positve_coverage, positve_size, \
        negative_coverage, negative_size, \
        maximum_coverage, maximum_size, \
        overall_coverage, overall_size, \
        coverage_intersection, size_intersection, \
        coverage_union, size_union, \
        coverage_maximum, size_maximum = compute(df, random_seed)

        positve_coverages.append(positve_coverage)
        positve_sizes.append(positve_size)
        negative_coverages.append(negative_coverage)
        negative_sizes.append(negative_size)
        overall_coverages.append(overall_coverage)
        overall_sizes.append(overall_size)
        maximum_coverages.append(maximum_coverage)
        maximum_sizes.append(maximum_size)
        coverages_intersection.append(coverage_intersection)
        sizes_intersection.append(size_intersection)
        coverages_union.append(coverage_union)
        sizes_union.append(size_union)
        coverages_maximum.append(coverage_maximum)
        sizes_maximum.append(size_maximum)
        
    print('average coverage of positive', np.mean(positve_coverages))
    print('average size of positive', np.mean(positve_sizes))
    print('average coverage of negative', np.mean(negative_coverages))
    print('average size of negative', np.mean(negative_sizes))
    print('average coverage of overall', np.mean(overall_coverages))
    print('average size of overall', np.mean(overall_sizes))
    print('average coverage of maximum', np.mean(maximum_coverages))
    print('average size of maximum', np.mean(maximum_sizes))
    print('average coverage of intersection', np.mean(coverages_intersection))
    print('average size of intersection', np.mean(sizes_intersection))
    print('average coverage of union', np.mean(coverages_union))
    print('average size of union', np.mean(sizes_union))
    print('average coverage of maximum', np.mean(coverages_maximum))
    print('average size of maximum', np.mean(sizes_maximum))
    print()
                  