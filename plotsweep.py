import torch
import sys
import plotting_functions
if __name__ == '__main__':
    #pathnamelist = ['reg0.0001','reg0.0005','reg0.005','reg0.001','reg1e-5','reg5e-5']
    pathnamelist = ['alternating']
    for pathname in pathnamelist:
        ####################
        # Complexity swipe #
        ###################
        #plotting_functions.HeatMapBVL('Hidden layer depth','Hidden layer size','Hidden layer size vs depth Heat Map',save_name=pathname + '_heatmap.png', HeatMap_dir='models/'+pathname,feature_1_name='linear',feature_2_name='linear_unit')

        ###############################
        # Lorentzian ratio and weight #
        ###############################
        #plotting_functions.HeatMapBVL('Lor_ratio','Lor_weight','Facilitated supervision weight and ratio Heat Map',save_name=pathname + '_heatmap.png', HeatMap_dir='models/'+pathname,feature_1_name='lor_ratio',feature_2_name='lor_weight')
        
        ###############
        # alternating #
        ###############
        #plotting_functions.HeatMapBVL('best validation loss','Lor_step','alternating training lor step Heat Map',save_name=pathname + '_heatmap.png', HeatMap_dir='models/'+pathname,feature_1_name='train_lor_step',feature_2_name=None)
        ##################
        # Lor train step #
        ##################
        plotting_functions.HeatMapBVL('Lor_ratio','Lor_step','Facilitated supervision weight and ratio Heat Map',save_name=pathname + '_heatmap.png', HeatMap_dir='models/'+pathname,feature_1_name='lor_ratio',feature_2_name='train_lor_step')
