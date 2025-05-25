import matplotlib.pyplot as plt
import re

# Paste your data as a multiline string
data = """
Trial 0 finished with value: 0.51417 | BEST! 0.51417 | Features: 29 | Params: boosting_type=goss, learning_rate=0.019527, num_leaves=18, max_depth=5, feature_fraction=0.777994                                             
Trial 1 finished with value: 3.04298 | Best: 0.51417 | Features: 29 | Params: boosting_type=dart, bagging_fraction=0.822433, bagging_freq=7, learning_rate=0.015561, num_leaves=12                                          
Trial 2 finished with value: 0.55996 | Best: 0.51417 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.818616, bagging_freq=7, learning_rate=0.012341, num_leaves=12                                          
Trial 3 finished with value: 0.53091 | Best: 0.51417 | Features: 35 | Params: boosting_type=goss, learning_rate=0.039873, num_leaves=19, max_depth=3, feature_fraction=0.821109                                             
Trial 4 finished with value: 0.51326 | BEST! 0.51326 | Features: 32 | Params: boosting_type=goss, learning_rate=0.024235, num_leaves=16, max_depth=4, feature_fraction=0.873880                                             
Trial 5 finished with value: 1.01716 | Best: 0.51326 | Features: 34 | Params: boosting_type=dart, bagging_fraction=0.831844, bagging_freq=1, learning_rate=0.035474, num_leaves=19                                          
Trial 6 finished with value: 1.21567 | Best: 0.51326 | Features: 34 | Params: boosting_type=dart, bagging_fraction=0.931248, bagging_freq=2, learning_rate=0.014973, num_leaves=25                                          
Trial 7 finished with value: 0.51441 | Best: 0.51326 | Features: 30 | Params: boosting_type=gbdt, bagging_fraction=0.990907, bagging_freq=4, learning_rate=0.024937, num_leaves=21                                          
Trial 8 finished with value: 1.21702 | Best: 0.51326 | Features: 30 | Params: boosting_type=goss, learning_rate=0.015217, num_leaves=16, max_depth=3, feature_fraction=0.881615                                             
Trial 9 finished with value: 1.00835 | Best: 0.51326 | Features: 27 | Params: boosting_type=dart, bagging_fraction=0.843860, bagging_freq=2, learning_rate=0.037947, num_leaves=29                                          
Trial 10 finished with value: 0.58063 | Best: 0.51326 | Features: 29 | Params: boosting_type=goss, learning_rate=0.026366, num_leaves=8, max_depth=6, feature_fraction=0.706361                                             
Trial 11 finished with value: 1.21141 | Best: 0.51326 | Features: 29 | Params: boosting_type=goss, learning_rate=0.020428, num_leaves=15, max_depth=6, feature_fraction=0.783749                                            
Trial 12 finished with value: 1.20983 | Best: 0.51326 | Features: 30 | Params: boosting_type=goss, learning_rate=0.021514, num_leaves=24, max_depth=5, feature_fraction=0.783780                                            
Trial 13 finished with value: 0.51420 | Best: 0.51326 | Features: 32 | Params: boosting_type=goss, learning_rate=0.030315, num_leaves=15, max_depth=5, feature_fraction=0.897644                                            
Trial 14 finished with value: 0.51173 | BEST! 0.51173 | Features: 28 | Params: boosting_type=goss, learning_rate=0.047742, num_leaves=31, max_depth=4, feature_fraction=0.800050                                            
Trial 15 finished with value: 0.51361 | Best: 0.51173 | Features: 27 | Params: boosting_type=goss, learning_rate=0.045312, num_leaves=30, max_depth=4, feature_fraction=0.864963                                            
Trial 16 finished with value: 0.51268 | Best: 0.51173 | Features: 27 | Params: boosting_type=gbdt, bagging_fraction=0.896301, bagging_freq=5, learning_rate=0.048565, num_leaves=32                                         
Trial 17 finished with value: 0.55079 | Best: 0.51173 | Features: 28 | Params: boosting_type=gbdt, bagging_fraction=0.898236, bagging_freq=5, learning_rate=0.047975, num_leaves=32                                         
Trial 18 finished with value: 0.50977 | BEST! 0.50977 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.898091, bagging_freq=5, learning_rate=0.049624, num_leaves=27                                         
Trial 19 finished with value: 1.20406 | Best: 0.50977 | Features: 32 | Params: boosting_type=gbdt, bagging_fraction=0.950445, bagging_freq=4, learning_rate=0.031280, num_leaves=27                                         
Trial 20 finished with value: 0.51038 | Best: 0.50977 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.872976, bagging_freq=6, learning_rate=0.041661, num_leaves=23                                         
Trial 21 finished with value: 0.51060 | Best: 0.50977 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.869283, bagging_freq=6, learning_rate=0.041666, num_leaves=23                                         
Trial 22 finished with value: 0.51022 | Best: 0.50977 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.869269, bagging_freq=6, learning_rate=0.042145, num_leaves=23                                         
Trial 23 finished with value: 1.20044 | Best: 0.50977 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.865190, bagging_freq=6, learning_rate=0.033631, num_leaves=27                                         
Trial 24 finished with value: 0.50216 | BEST! 0.50216 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.874254, bagging_freq=5, learning_rate=0.042371, num_leaves=22                                         
Trial 25 finished with value: 1.20240 | Best: 0.50216 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.919125, bagging_freq=4, learning_rate=0.029122, num_leaves=27                                         
Trial 26 finished with value: 1.19836 | Best: 0.50216 | Features: 33 | Params: boosting_type=gbdt, bagging_fraction=0.886225, bagging_freq=5, learning_rate=0.034520, num_leaves=21                                         
Trial 27 finished with value: 0.50107 | BEST! 0.50107 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.848602, bagging_freq=5, learning_rate=0.042843, num_leaves=25                                         
Trial 28 finished with value: 1.19467 | Best: 0.50107 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.854762, bagging_freq=3, learning_rate=0.038483, num_leaves=26                                         
Trial 29 finished with value: 1.21343 | Best: 0.50107 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.915894, bagging_freq=5, learning_rate=0.017486, num_leaves=29                                         
Trial 30 finished with value: 0.49956 | BEST! 0.49956 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.805279, bagging_freq=3, learning_rate=0.049747, num_leaves=21                                         
Trial 31 finished with value: 0.50108 | Best: 0.49956 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.800257, bagging_freq=3, learning_rate=0.044102, num_leaves=21                                         
Trial 32 finished with value: 1.22044 | Best: 0.49956 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.803504, bagging_freq=3, learning_rate=0.010077, num_leaves=21                                         
Trial 33 finished with value: 0.50399 | Best: 0.49956 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.800063, bagging_freq=3, learning_rate=0.043280, num_leaves=18                                         
Trial 34 finished with value: 1.19675 | Best: 0.49956 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.837903, bagging_freq=4, learning_rate=0.036564, num_leaves=20                                         
Trial 35 finished with value: 0.49931 | BEST! 0.49931 | Features: 29 | Params: boosting_type=gbdt, bagging_fraction=0.822834, bagging_freq=2, learning_rate=0.044435, num_leaves=22                                         
Trial 36 finished with value: 1.19962 | Best: 0.49931 | Features: 29 | Params: boosting_type=dart, bagging_fraction=0.813114, bagging_freq=2, learning_rate=0.032411, num_leaves=25                                         
Trial 37 finished with value: 0.50331 | Best: 0.49931 | Features: 30 | Params: boosting_type=gbdt, bagging_fraction=0.827356, bagging_freq=1, learning_rate=0.045936, num_leaves=18                                         
Trial 38 finished with value: 1.19544 | Best: 0.49931 | Features: 27 | Params: boosting_type=dart, bagging_fraction=0.811120, bagging_freq=3, learning_rate=0.038462, num_leaves=20                                         
Trial 39 finished with value: 1.20297 | Best: 0.49931 | Features: 29 | Params: boosting_type=gbdt, bagging_fraction=0.849717, bagging_freq=2, learning_rate=0.028059, num_leaves=25                                         
Trial 40 finished with value: 1.19941 | Best: 0.49931 | Features: 30 | Params: boosting_type=gbdt, bagging_fraction=0.824447, bagging_freq=3, learning_rate=0.035976, num_leaves=13                                         
Trial 41 finished with value: 0.50187 | Best: 0.49931 | Features: 29 | Params: boosting_type=gbdt, bagging_fraction=0.820572, bagging_freq=2, learning_rate=0.043964, num_leaves=22                                         
Trial 42 finished with value: 0.50461 | Best: 0.49931 | Features: 29 | Params: boosting_type=gbdt, bagging_fraction=0.816147, bagging_freq=2, learning_rate=0.045334, num_leaves=18                                         
Trial 43 finished with value: 1.19340 | Best: 0.49931 | Features: 29 | Params: boosting_type=gbdt, bagging_fraction=0.801594, bagging_freq=1, learning_rate=0.039655, num_leaves=22                                         
Trial 44 finished with value: 0.49882 | BEST! 0.49882 | Features: 30 | Params: boosting_type=gbdt, bagging_fraction=0.829714, bagging_freq=3, learning_rate=0.044393, num_leaves=24                                         
Trial 45 finished with value: 0.96115 | Best: 0.49882 | Features: 29 | Params: boosting_type=dart, bagging_fraction=0.835256, bagging_freq=3, learning_rate=0.049936, num_leaves=24                                         
Trial 46 finished with value: 1.19355 | Best: 0.49882 | Features: 34 | Params: boosting_type=gbdt, bagging_fraction=0.830839, bagging_freq=4, learning_rate=0.040169, num_leaves=19                                         
Trial 47 finished with value: 1.21749 | Best: 0.49882 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.809777, bagging_freq=3, learning_rate=0.013206, num_leaves=24                                         
Trial 48 finished with value: 1.21323 | Best: 0.49882 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.840759, bagging_freq=2, learning_rate=0.018493, num_leaves=17                                         
Trial 49 finished with value: 1.02827 | Best: 0.49882 | Features: 32 | Params: boosting_type=dart, bagging_fraction=0.855973, bagging_freq=4, learning_rate=0.046131, num_leaves=21                                         
Trial 50 finished with value: 1.19687 | Best: 0.49882 | Features: 33 | Params: boosting_type=gbdt, bagging_fraction=0.823615, bagging_freq=3, learning_rate=0.035777, num_leaves=26                                         
Trial 51 finished with value: 0.50207 | Best: 0.49882 | Features: 29 | Params: boosting_type=gbdt, bagging_fraction=0.817701, bagging_freq=2, learning_rate=0.044131, num_leaves=22                                         
Trial 52 finished with value: 0.50010 | Best: 0.49882 | Features: 30 | Params: boosting_type=gbdt, bagging_fraction=0.822353, bagging_freq=1, learning_rate=0.049832, num_leaves=23                                         
Trial 53 finished with value: 0.49959 | Best: 0.49882 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.843866, bagging_freq=1, learning_rate=0.049593, num_leaves=20                                         
Trial 54 finished with value: 0.50232 | Best: 0.49882 | Features: 31 | Params: boosting_type=gbdt, bagging_fraction=0.841167, bagging_freq=1, learning_rate=0.049910, num_leaves=19                                         
Trial 55 finished with value: 0.49752 | BEST! 0.49752 | Features: 32 | Params: boosting_type=goss, learning_rate=0.047285, num_leaves=24, max_depth=6, feature_fraction=0.835967                                            
Trial 56 finished with value: 0.55321 | Best: 0.49752 | Features: 30 | Params: boosting_type=goss, learning_rate=0.046977, num_leaves=23, max_depth=6, feature_fraction=0.814142                                            
Trial 57 finished with value: 0.49543 | BEST! 0.49543 | Features: 27 | Params: boosting_type=goss, learning_rate=0.047184, num_leaves=20, max_depth=6, feature_fraction=0.842337                                            
Trial 58 finished with value: 1.20944 | Best: 0.49543 | Features: 27 | Params: boosting_type=goss, learning_rate=0.024250, num_leaves=8, max_depth=6, feature_fraction=0.830700                                             
Trial 59 finished with value: 1.20970 | Best: 0.49543 | Features: 28 | Params: boosting_type=goss, learning_rate=0.022756, num_leaves=16, max_depth=6, feature_fraction=0.881175                                            
Trial 60 finished with value: 1.19312 | Best: 0.49543 | Features: 30 | Params: boosting_type=goss, learning_rate=0.040318, num_leaves=20, max_depth=6, feature_fraction=0.992010                                            
Trial 61 finished with value: 0.49319 | BEST! 0.49319 | Features: 27 | Params: boosting_type=goss, learning_rate=0.047767, num_leaves=24, max_depth=6, feature_fraction=0.835925                                            
Trial 62 finished with value: 0.49303 | BEST! 0.49303 | Features: 27 | Params: boosting_type=goss, learning_rate=0.047089, num_leaves=24, max_depth=6, feature_fraction=0.839669                                            
Trial 63 finished with value: 0.49213 | BEST! 0.49213 | Features: 27 | Params: boosting_type=goss, learning_rate=0.046915, num_leaves=26, max_depth=6, feature_fraction=0.835953                                            
Trial 64 finished with value: 0.49172 | BEST! 0.49172 | Features: 28 | Params: boosting_type=goss, learning_rate=0.046010, num_leaves=28, max_depth=6, feature_fraction=0.838418                                            
Trial 65 finished with value: 0.49108 | BEST! 0.49108 | Features: 29 | Params: boosting_type=goss, learning_rate=0.046639, num_leaves=28, max_depth=6, feature_fraction=0.835238                                            
Trial 66 finished with value: 1.19114 | Best: 0.49108 | Features: 28 | Params: boosting_type=goss, learning_rate=0.041010, num_leaves=29, max_depth=6, feature_fraction=0.834539                                            
Trial 67 finished with value: 0.49578 | Best: 0.49108 | Features: 29 | Params: boosting_type=goss, learning_rate=0.046763, num_leaves=28, max_depth=6, feature_fraction=0.850703                                            
Trial 68 finished with value: 1.19483 | Best: 0.49108 | Features: 33 | Params: boosting_type=goss, learning_rate=0.037081, num_leaves=28, max_depth=6, feature_fraction=0.851179                                            
Trial 69 finished with value: 0.53755 | Best: 0.49108 | Features: 25 | Params: boosting_type=goss, learning_rate=0.047090, num_leaves=30, max_depth=6, feature_fraction=0.869849                                            
Trial 70 finished with value: 1.19291 | Best: 0.49108 | Features: 30 | Params: boosting_type=goss, learning_rate=0.039252, num_leaves=28, max_depth=6, feature_fraction=0.895631                                            
Trial 71 finished with value: 0.49634 | Best: 0.49108 | Features: 29 | Params: boosting_type=goss, learning_rate=0.046767, num_leaves=26, max_depth=6, feature_fraction=0.846527                                            
Trial 72 finished with value: 1.19034 | Best: 0.49108 | Features: 29 | Params: boosting_type=goss, learning_rate=0.042445, num_leaves=26, max_depth=6, feature_fraction=0.859486                                            
Trial 73 finished with value: 0.49637 | Best: 0.49108 | Features: 29 | Params: boosting_type=goss, learning_rate=0.045363, num_leaves=28, max_depth=6, feature_fraction=0.844507                                            
Trial 74 finished with value: 0.49492 | Best: 0.49108 | Features: 29 | Params: boosting_type=goss, learning_rate=0.047445, num_leaves=30, max_depth=6, feature_fraction=0.850134                                            
Trial 75 finished with value: 1.19152 | Best: 0.49108 | Features: 29 | Params: boosting_type=goss, learning_rate=0.041477, num_leaves=30, max_depth=6, feature_fraction=0.877448                                            
Trial 76 finished with value: 0.49576 | Best: 0.49108 | Features: 26 | Params: boosting_type=goss, learning_rate=0.046936, num_leaves=31, max_depth=6, feature_fraction=0.830520                                            
Trial 77 finished with value: 0.49526 | Best: 0.49108 | Features: 26 | Params: boosting_type=goss, learning_rate=0.047994, num_leaves=32, max_depth=6, feature_fraction=0.828411                                            
Trial 78 finished with value: 1.19443 | Best: 0.49108 | Features: 26 | Params: boosting_type=goss, learning_rate=0.037539, num_leaves=31, max_depth=6, feature_fraction=0.889857                                            
Trial 79 finished with value: 1.19141 | Best: 0.49108 | Features: 28 | Params: boosting_type=goss, learning_rate=0.042883, num_leaves=32, max_depth=3, feature_fraction=0.864071                                            
Trial 80 finished with value: 0.48803 | BEST! 0.48803 | Features: 28 | Params: boosting_type=goss, learning_rate=0.047934, num_leaves=29, max_depth=6, feature_fraction=0.822354                                            
Trial 81 finished with value: 0.48714 | BEST! 0.48714 | Features: 28 | Params: boosting_type=goss, learning_rate=0.048145, num_leaves=30, max_depth=6, feature_fraction=0.825297                                            
Trial 82 finished with value: 0.48840 | Best: 0.48714 | Features: 28 | Params: boosting_type=goss, learning_rate=0.047954, num_leaves=29, max_depth=6, feature_fraction=0.824655                                            
Trial 83 finished with value: 1.18830 | Best: 0.48714 | Features: 28 | Params: boosting_type=goss, learning_rate=0.044782, num_leaves=29, max_depth=6, feature_fraction=0.818209                                            
Trial 84 finished with value: 1.19116 | Best: 0.48714 | Features: 28 | Params: boosting_type=goss, learning_rate=0.041775, num_leaves=30, max_depth=6, feature_fraction=0.810435                                            
Trial 85 finished with value: 1.19800 | Best: 0.48714 | Features: 28 | Params: boosting_type=goss, learning_rate=0.034099, num_leaves=27, max_depth=6, feature_fraction=0.795421                                            
Trial 86 finished with value: 0.48771 | Best: 0.48714 | Features: 30 | Params: boosting_type=goss, learning_rate=0.048385, num_leaves=29, max_depth=6, feature_fraction=0.822970                                            
Trial 87 finished with value: 1.18993 | Best: 0.48714 | Features: 30 | Params: boosting_type=goss, learning_rate=0.043570, num_leaves=29, max_depth=6, feature_fraction=0.822567                                            
Trial 88 finished with value: 1.21430 | Best: 0.48714 | Features: 28 | Params: boosting_type=goss, learning_rate=0.015989, num_leaves=31, max_depth=6, feature_fraction=0.837362                                            
Trial 89 finished with value: 0.48754 | Best: 0.48714 | Features: 31 | Params: boosting_type=goss, learning_rate=0.048782, num_leaves=28, max_depth=6, feature_fraction=0.825678                                            
Trial 90 finished with value: 1.18890 | Best: 0.48714 | Features: 31 | Params: boosting_type=goss, learning_rate=0.045161, num_leaves=27, max_depth=6, feature_fraction=0.799132                                            
Trial 91 finished with value: 0.48737 | Best: 0.48714 | Features: 31 | Params: boosting_type=goss, learning_rate=0.048571, num_leaves=29, max_depth=6, feature_fraction=0.825896                                            
Trial 92 finished with value: 0.48793 | Best: 0.48714 | Features: 31 | Params: boosting_type=goss, learning_rate=0.048686, num_leaves=28, max_depth=6, feature_fraction=0.826071                                            
Trial 93 finished with value: 0.48780 | Best: 0.48714 | Features: 31 | Params: boosting_type=goss, learning_rate=0.049532, num_leaves=28, max_depth=6, feature_fraction=0.823374                                            
Trial 94 finished with value: 0.48741 | Best: 0.48714 | Features: 31 | Params: boosting_type=goss, learning_rate=0.048908, num_leaves=29, max_depth=6, feature_fraction=0.824823                                            
Trial 95 finished with value: 0.48730 | Best: 0.48714 | Features: 31 | Params: boosting_type=goss, learning_rate=0.048658, num_leaves=29, max_depth=6, feature_fraction=0.809346                                            
Trial 96 finished with value: 0.50734 | Best: 0.48714 | Features: 33 | Params: boosting_type=goss, learning_rate=0.048825, num_leaves=29, max_depth=6, feature_fraction=0.809722                                            
Trial 97 finished with value: 0.48769 | Best: 0.48714 | Features: 30 | Params: boosting_type=goss, learning_rate=0.048778, num_leaves=31, max_depth=6, feature_fraction=0.822566                                            
Trial 98 finished with value: 1.01076 | Best: 0.48714 | Features: 31 | Params: boosting_type=dart, bagging_fraction=0.983197, bagging_freq=7, learning_rate=0.049727, num_leaves=30                                         
Trial 99 finished with value: 1.18740 | Best: 0.48714 | Features: 30 | Params: boosting_type=goss, learning_rate=0.044002, num_leaves=31, max_depth=6, feature_fraction=0.816114                                            
Trial 100 finished with value: 1.21982 | Best: 0.48714 | Features: 31 | Params: boosting_type=goss, learning_rate=0.010432, num_leaves=30, max_depth=6, feature_fraction=0.802913                                           
Trial 101 finished with value: 0.48849 | Best: 0.48714 | Features: 30 | Params: boosting_type=goss, learning_rate=0.048480, num_leaves=29, max_depth=6, feature_fraction=0.826536                                           
Trial 102 finished with value: 0.48845 | Best: 0.48714 | Features: 30 | Params: boosting_type=goss, learning_rate=0.048621, num_leaves=29, max_depth=6, feature_fraction=0.820952                                           
Trial 103 finished with value: 1.14801 | Best: 0.48714 | Features: 30 | Params: boosting_type=goss, learning_rate=0.044924, num_leaves=31, max_depth=6, feature_fraction=0.810188                                           
Trial 104 finished with value: 1.19017 | Best: 0.48714 | Features: 31 | Params: boosting_type=goss, learning_rate=0.042626, num_leaves=28, max_depth=6, feature_fraction=0.826134                                           
Trial 105 finished with value: 0.48777 | Best: 0.48714 | Features: 31 | Params: boosting_type=goss, learning_rate=0.048709, num_leaves=27, max_depth=6, feature_fraction=0.799037                                           
Trial 106 finished with value: 0.48660 | BEST! 0.48660 | Features: 30 | Params: boosting_type=goss, learning_rate=0.049768, num_leaves=27, max_depth=6, feature_fraction=0.802694                                           
Trial 107 finished with value: 0.49229 | Best: 0.48660 | Features: 29 | Params: boosting_type=goss, learning_rate=0.049828, num_leaves=27, max_depth=6, feature_fraction=0.796394                                           
Trial 108 finished with value: 1.14794 | Best: 0.48660 | Features: 28 | Params: boosting_type=goss, learning_rate=0.045487, num_leaves=28, max_depth=6, feature_fraction=0.781958                                           
Trial 109 finished with value: 1.18919 | Best: 0.48660 | Features: 31 | Params: boosting_type=goss, learning_rate=0.043496, num_leaves=27, max_depth=6, feature_fraction=0.802999                                           
Trial 110 finished with value: 1.19171 | Best: 0.48660 | Features: 33 | Params: boosting_type=dart, bagging_fraction=0.958333, bagging_freq=7, learning_rate=0.040709, num_leaves=32                                        
Trial 111 finished with value: 0.48660 | BEST! 0.48660 | Features: 30 | Params: boosting_type=goss, learning_rate=0.048494, num_leaves=30, max_depth=6, feature_fraction=0.806439                                           
Trial 112 finished with value: 1.18837 | Best: 0.48660 | Features: 30 | Params: boosting_type=goss, learning_rate=0.045684, num_leaves=30, max_depth=6, feature_fraction=0.791064                                           
Trial 113 finished with value: 0.48679 | Best: 0.48660 | Features: 30 | Params: boosting_type=goss, learning_rate=0.049948, num_leaves=28, max_depth=6, feature_fraction=0.806179                                           
Trial 114 finished with value: 1.20465 | Best: 0.48660 | Features: 30 | Params: boosting_type=goss, learning_rate=0.026269, num_leaves=30, max_depth=6, feature_fraction=0.809438                                           
Trial 115 finished with value: 0.48586 | BEST! 0.48586 | Features: 30 | Params: boosting_type=goss, learning_rate=0.049919, num_leaves=31, max_depth=6, feature_fraction=0.804641                                           
Trial 116 finished with value: 1.18760 | Best: 0.48586 | Features: 30 | Params: boosting_type=goss, learning_rate=0.045700, num_leaves=31, max_depth=6, feature_fraction=0.796404                                           
Trial 117 finished with value: 1.18753 | Best: 0.48586 | Features: 30 | Params: boosting_type=goss, learning_rate=0.043700, num_leaves=32, max_depth=6, feature_fraction=0.802078                                           
Trial 118 finished with value: 0.48667 | Best: 0.48586 | Features: 30 | Params: boosting_type=goss, learning_rate=0.048760, num_leaves=30, max_depth=6, feature_fraction=0.783361                                           
Trial 119 finished with value: 1.20979 | Best: 0.48586 | Features: 30 | Params: boosting_type=goss, learning_rate=0.020327, num_leaves=30, max_depth=6, feature_fraction=0.963941                                           
Trial 120 finished with value: 1.19412 | Best: 0.48586 | Features: 30 | Params: boosting_type=goss, learning_rate=0.039047, num_leaves=31, max_depth=6, feature_fraction=0.786356                                           
Trial 121 finished with value: 0.48673 | Best: 0.48586 | Features: 30 | Params: boosting_type=goss, learning_rate=0.048405, num_leaves=29, max_depth=6, feature_fraction=0.779465                                           
Trial 122 finished with value: 1.18803 | Best: 0.48586 | Features: 30 | Params: boosting_type=goss, learning_rate=0.046024, num_leaves=31, max_depth=6, feature_fraction=0.779260                                           
Trial 123 finished with value: 1.19004 | Best: 0.48586 | Features: 30 | Params: boosting_type=goss, learning_rate=0.049911, num_leaves=30, max_depth=3, feature_fraction=0.813920                                           
Trial 124 finished with value: 0.48653 | Best: 0.48586 | Features: 30 | Params: boosting_type=goss, learning_rate=0.048049, num_leaves=29, max_depth=6, feature_fraction=0.805375                                           
Trial 125 finished with value: 1.18929 | Best: 0.48586 | Features: 30 | Params: boosting_type=goss, learning_rate=0.044575, num_leaves=29, max_depth=6, feature_fraction=0.791974                                           
Trial 126 finished with value: 0.48717 | Best: 0.48586 | Features: 30 | Params: boosting_type=goss, learning_rate=0.046571, num_leaves=30, max_depth=6, feature_fraction=0.804443                                           
Trial 127 finished with value: 1.18643 | Best: 0.48586 | Features: 27 | Params: boosting_type=goss, learning_rate=0.046440, num_leaves=30, max_depth=6, feature_fraction=0.807102                                           
Trial 128 finished with value: 1.18990 | Best: 0.48586 | Features: 30 | Params: boosting_type=goss, learning_rate=0.043151, num_leaves=29, max_depth=6, feature_fraction=0.759544                                           
Trial 129 finished with value: 0.48762 | Best: 0.48586 | Features: 29 | Params: boosting_type=goss, learning_rate=0.047228, num_leaves=28, max_depth=6, feature_fraction=0.777108                                           
Trial 130 finished with value: 1.00863 | Best: 0.48586 | Features: 30 | Params: boosting_type=dart, bagging_fraction=0.946440, bagging_freq=6, learning_rate=0.049982, num_leaves=30                                        
Trial 131 finished with value: 0.48764 | Best: 0.48586 | Features: 29 | Params: boosting_type=goss, learning_rate=0.047482, num_leaves=28, max_depth=6, feature_fraction=0.776703                                           
Trial 132 finished with value: 1.18645 | Best: 0.48586 | Features: 29 | Params: boosting_type=goss, learning_rate=0.045707, num_leaves=29, max_depth=6, feature_fraction=0.792163                                           
Trial 133 finished with value: 0.48702 | Best: 0.48586 | Features: 29 | Params: boosting_type=goss, learning_rate=0.047768, num_leaves=28, max_depth=6, feature_fraction=0.782087                                           
Trial 134 finished with value: 1.18737 | Best: 0.48586 | Features: 29 | Params: boosting_type=goss, learning_rate=0.044642, num_leaves=30, max_depth=6, feature_fraction=0.802414                                           
Trial 135 finished with value: 1.19101 | Best: 0.48586 | Features: 29 | Params: boosting_type=goss, learning_rate=0.041726, num_leaves=29, max_depth=6, feature_fraction=0.807419                                           
Trial 136 finished with value: 0.48696 | Best: 0.48586 | Features: 31 | Params: boosting_type=goss, learning_rate=0.048021, num_leaves=27, max_depth=6, feature_fraction=0.816686                                           
Trial 137 finished with value: 0.51767 | Best: 0.48586 | Features: 33 | Params: boosting_type=goss, learning_rate=0.047767, num_leaves=27, max_depth=6, feature_fraction=0.815087                                           
Trial 138 finished with value: 1.18746 | Best: 0.48586 | Features: 30 | Params: boosting_type=goss, learning_rate=0.045857, num_leaves=26, max_depth=6, feature_fraction=0.783646                                           
Trial 139 finished with value: 0.48624 | Best: 0.48586 | Features: 31 | Params: boosting_type=goss, learning_rate=0.049921, num_leaves=29, max_depth=6, feature_fraction=0.796024                                           
Trial 140 finished with value: 0.48615 | Best: 0.48586 | Features: 31 | Params: boosting_type=goss, learning_rate=0.046828, num_leaves=32, max_depth=6, feature_fraction=0.794228                                           
Trial 141 finished with value: 0.48616 | Best: 0.48586 | Features: 31 | Params: boosting_type=goss, learning_rate=0.046825, num_leaves=31, max_depth=6, feature_fraction=0.795677                                           
Trial 142 finished with value: 0.48532 | BEST! 0.48532 | Features: 31 | Params: boosting_type=goss, learning_rate=0.049912, num_leaves=32, max_depth=6, feature_fraction=0.795303                                           
Trial 143 finished with value: 0.48566 | Best: 0.48532 | Features: 31 | Params: boosting_type=goss, learning_rate=0.046765, num_leaves=32, max_depth=6, feature_fraction=0.769296                                           
Trial 144 finished with value: 0.48537 | Best: 0.48532 | Features: 31 | Params: boosting_type=goss, learning_rate=0.049897, num_leaves=32, max_depth=6, feature_fraction=0.788693                                           
Trial 145 finished with value: 0.48548 | Best: 0.48532 | Features: 31 | Params: boosting_type=goss, learning_rate=0.049996, num_leaves=32, max_depth=6, feature_fraction=0.770652                                           
Trial 146 finished with value: 0.49134 | Best: 0.48532 | Features: 30 | Params: boosting_type=goss, learning_rate=0.049978, num_leaves=32, max_depth=6, feature_fraction=0.768892                                           
Trial 147 finished with value: 1.18725 | Best: 0.48532 | Features: 32 | Params: boosting_type=goss, learning_rate=0.043985, num_leaves=32, max_depth=6, feature_fraction=0.758301                                           
Trial 148 finished with value: 1.18607 | Best: 0.48532 | Features: 31 | Params: boosting_type=goss, learning_rate=0.046010, num_leaves=32, max_depth=6, feature_fraction=0.766804                                           
Trial 149 finished with value: 1.18607 | Best: 0.48532 | Features: 33 | Params: boosting_type=goss, learning_rate=0.046774, num_leaves=32, max_depth=6, feature_fraction=0.773330                                           
Trial 150 finished with value: 1.18835 | Best: 0.48532 | Features: 29 | Params: boosting_type=goss, learning_rate=0.042436, num_leaves=31, max_depth=6, feature_fraction=0.747182                                           
Trial 151 finished with value: 0.48540 | Best: 0.48532 | Features: 31 | Params: boosting_type=goss, learning_rate=0.049997, num_leaves=31, max_depth=6, feature_fraction=0.783649                                           
Trial 152 finished with value: 0.48498 | BEST! 0.48498 | Features: 31 | Params: boosting_type=goss, learning_rate=0.049959, num_leaves=32, max_depth=6, feature_fraction=0.794770                                           
Trial 153 finished with value: 0.48577 | Best: 0.48498 | Features: 31 | Params: boosting_type=goss, learning_rate=0.049923, num_leaves=32, max_depth=6, feature_fraction=0.794158                                           
Trial 154 finished with value: 0.48590 | Best: 0.48498 | Features: 31 | Params: boosting_type=goss, learning_rate=0.049992, num_leaves=32, max_depth=6, feature_fraction=0.791580                                           
"""

# Parse the data and keep only runs where the score is the best so far
trial_nums = []
values = []
n_features = []

best_so_far = float('inf')
for line in data.strip().split('\n'):
    m = re.search(r'Trial (\d+) finished with value: ([\d\.]+)', line)
    f = re.search(r'Features: (\d+)', line)
    if m and f:
        value = float(m.group(2))
        if value < best_so_far:
            trial_nums.append(int(m.group(1)))
            values.append(value)
            n_features.append(int(f.group(1)))
            best_so_far = value

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Trial')
ax1.set_ylabel('Value (Score)', color=color)
ax1.plot(trial_nums, values, marker='o', color=color, label='Best Score')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(bottom=0)

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Number of Features', color=color)
ax2.plot(trial_nums, n_features, marker='x', color=color, label='Num Features')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Optuna Trials: Best Scores and Number of Features')
fig.tight_layout()
plt.savefig("optuna_best_scores.png")
plt.close()