# config = {'datapath':'./work/stage1/preprocess/val/',
#           'preprocess_result_path':'./prep_result/',
#           'outputfile':'prediction.csv',
          
#           'detector_model':'net_detector',
#          'detector_param':'./model/detector.ckpt',
#          'classifier_model':'net_classifier',
#          'classifier_param':'./model/classifier.ckpt',
#          'n_gpu':2,
#          'n_worker_preprocessing':None,
#          'use_exsiting_preprocessing':False,
#          'skip_preprocessing':True,
#          'skip_detect':False}
img_process = "2Preprocessing_img/"

config = {'datapath':'../Image/2Preprocessing_img/',
          'preprocess_result_path':'../DSB2017/prep_result/',
          'outputfile':'prediction.csv',
          
          'detector_model':'net_detector',
         'detector_param':'../DSB2017/model/detector.ckpt',
         'bbox_result' : '../Image/3Detection_img/',
         'classifier_model':'net_classifier',
         'classifier_param':'../DSB2017/model/classifier.ckpt',
         'n_gpu':2,
         'n_worker_preprocessing':None,
         'use_exsiting_preprocessing':False,
         'skip_preprocessing':True,
         'skip_detect':False}