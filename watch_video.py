import shutil
import os

video_names = [
'v_OLCTPfweyk8_3_14'   , 'v_OLCTPfweyk8_15_34'  , 'v_OLCTPfweyk8_56_81'  , 'v_OLCTPfweyk8_81_101' , 'v_OLCTPfweyk8_101_113', 'v_OLCTPfweyk8_113_124', 
'v_OLCTPfweyk8_124_143', 'v_OLCTPfweyk8_143_162', 'v_OLCTPfweyk8_162_167', 'v_OLCTPfweyk8_167_180', 'v_OLCTPfweyk8_180_194', 'v_OLCTPfweyk8_240_259',
'v_OLCTPfweyk8_273_306', 'v_OLCTPfweyk8_306_322', 'v_OLCTPfweyk8_322_333', 'v_OLCTPfweyk8_333_342', 'v_OLCTPfweyk8_342_366', 'v_OLCTPfweyk8_366_373',
'v_OLCTPfweyk8_421_435', 'v_OLCTPfweyk8_435_455', 'v_OLCTPfweyk8_539_569', 'v_OLCTPfweyk8_569_595', 'v_GkbG1OHfSRw_3_14'   , 'v_GkbG1OHfSRw_32_41'  ,
'v_GkbG1OHfSRw_41_60'  , 'v_GkbG1OHfSRw_66_89'  , 'v_GkbG1OHfSRw_127_151', 'v_GkbG1OHfSRw_151_173', 'v_GkbG1OHfSRw_173_202', 'v_GkbG1OHfSRw_226_246',
'v_GkbG1OHfSRw_246_276', 'v_GkbG1OHfSRw_276_285', 'v_GkbG1OHfSRw_285_293', 'v_GkbG1OHfSRw_321_346', 'v_GkbG1OHfSRw_390_416', 'v_GkbG1OHfSRw_486_504', 
'v_GkbG1OHfSRw_504_540', 'v_CR_GII1VPAk_2_14'   , 'v_CR_GII1VPAk_18_32'  , 'v_CR_GII1VPAk_32_48'  , 'v_CR_GII1VPAk_57_70'  , 'v_CR_GII1VPAk_70_93'  , 
'v_CR_GII1VPAk_93_105' , 'v_CR_GII1VPAk_136_152', 'v_CR_GII1VPAk_152_176', 'v_CR_GII1VPAk_191_213', 'v_CR_GII1VPAk_213_231', 'v_CR_GII1VPAk_231_257', 
'v_CR_GII1VPAk_257_275', 'v_CR_GII1VPAk_275_297', 'v_CR_GII1VPAk_340_349', 'v_CR_GII1VPAk_353_373', 'v_aCQJOksAYag_2_22'   , 'v_aCQJOksAYag_22_52'  , 
'v_aCQJOksAYag_52_62'  , 'v_aCQJOksAYag_67_83'  , 'v_aCQJOksAYag_83_102' , 'v_aCQJOksAYag_102_120', 'v_aCQJOksAYag_120_135', 'v_aCQJOksAYag_135_148', 
'v_aCQJOksAYag_148_166', 'v_aCQJOksAYag_166_182', 'v_aCQJOksAYag_182_199', 'v_aCQJOksAYag_199_212', 'v_aCQJOksAYag_212_233', 'v_aCQJOksAYag_247_263', 
'v_aCQJOksAYag_263_286', 'v_aCQJOksAYag_286_307', 'v_aCQJOksAYag_322_336', 'v_aCQJOksAYag_336_349', 'v_aCQJOksAYag_349_367', 'v_aCQJOksAYag_368_384', 
'v_aCQJOksAYag_385_405', 'v_aCQJOksAYag_405_433', 'v_aCQJOksAYag_433_452', 'v_aCQJOksAYag_476_495', 'v_aCQJOksAYag_495_515', 'v_aCQJOksAYag_516_533', 
'v_WZIlgmTgvfE_3_22'   , 'v_WZIlgmTgvfE_22_63'  , 'v_WZIlgmTgvfE_63_87'  , 'v_WZIlgmTgvfE_128_144', 'v_WZIlgmTgvfE_144_158', 'v_WZIlgmTgvfE_158_176', 
'v_WZIlgmTgvfE_177_194', 'v_WZIlgmTgvfE_208_217', 'v_WZIlgmTgvfE_217_227', 'v_WZIlgmTgvfE_227_241', 'v_WZIlgmTgvfE_241_269', 'v_WZIlgmTgvfE_269_283', 
'v_WZIlgmTgvfE_294_306', 'v_WZIlgmTgvfE_306_318', 'v_WZIlgmTgvfE_318_340', 'v_WZIlgmTgvfE_340_360', 'v_WZIlgmTgvfE_360_375', 'v_WZIlgmTgvfE_375_385', 
'v_WZIlgmTgvfE_385_399', 'v_VoIgOqvr6f0_2_17'   , 'v_VoIgOqvr6f0_18_31'  , 'v_VoIgOqvr6f0_61_79'  , 'v_VoIgOqvr6f0_79_96'  , 'v_VoIgOqvr6f0_96_111' , 
'v_VoIgOqvr6f0_111_126', 'v_VoIgOqvr6f0_141_159', 'v_VoIgOqvr6f0_159_175', 'v_VoIgOqvr6f0_176_200', 'v_VoIgOqvr6f0_201_208', 'v_VoIgOqvr6f0_214_244', 
'v_VoIgOqvr6f0_244_260', 'v_VoIgOqvr6f0_260_275', 'v_VoIgOqvr6f0_275_286', 'v_VoIgOqvr6f0_286_302', 'v_VoIgOqvr6f0_302_321', 'v_VoIgOqvr6f0_321_336', 
'v_VoIgOqvr6f0_336_361', 'v_VoIgOqvr6f0_410_420', 'v_j0c3S0fRUok_14_28'  , 'v_j0c3S0fRUok_28_43'  , 'v_j0c3S0fRUok_43_60'  , 'v_j0c3S0fRUok_82_97'  , 
'v_j0c3S0fRUok_97_139' , 'v_j0c3S0fRUok_180_219', 'v_j0c3S0fRUok_219_254', 'v_j0c3S0fRUok_254_271', 'v_j0c3S0fRUok_285_303', 'v_j0c3S0fRUok_303_318', 
'v_KVkYkt4tTPE_26_41'  , 'v_KVkYkt4tTPE_41_55'  , 'v_KVkYkt4tTPE_74_89'  , 'v_KVkYkt4tTPE_167_187', 'v_KVkYkt4tTPE_187_198', 'v_KVkYkt4tTPE_217_234', 
'v_KVkYkt4tTPE_234_253', 'v_KVkYkt4tTPE_254_268', 'v_KVkYkt4tTPE_268_281', 'v_KVkYkt4tTPE_288_299', 'v_KVkYkt4tTPE_308_320', 'v_oFBpB9_ibaA_3_20'   , 
'v_oFBpB9_ibaA_20_32'  , 'v_oFBpB9_ibaA_32_48'  , 'v_oFBpB9_ibaA_48_64'  , 'v_oFBpB9_ibaA_79_118' , 'v_oFBpB9_ibaA_159_173', 'v_oFBpB9_ibaA_173_185', 
'v_oFBpB9_ibaA_206_228', 'v_oFBpB9_ibaA_229_236', 'v_oFBpB9_ibaA_237_241', 'v_oFBpB9_ibaA_248_268', 'v_oFBpB9_ibaA_268_279', 'v_oFBpB9_ibaA_308_321', 
'v_oFBpB9_ibaA_321_331', 'v_oFBpB9_ibaA_341_351', 'v_oFBpB9_ibaA_364_382', 'v_oFBpB9_ibaA_400_419', 'v_oFBpB9_ibaA_419_434', 'v_oFBpB9_ibaA_480_493', 
'v_oFBpB9_ibaA_576_597', 'v_hZ5hgZBmUzw_3_28'   , 'v_hZ5hgZBmUzw_28_45'  , 'v_hZ5hgZBmUzw_45_68'  , 'v_hZ5hgZBmUzw_120_140', 'v_hZ5hgZBmUzw_140_156', 
'v_hZ5hgZBmUzw_156_177', 'v_hZ5hgZBmUzw_177_204', 'v_hZ5hgZBmUzw_204_218', 'v_hZ5hgZBmUzw_218_232', 'v_hZ5hgZBmUzw_232_245', 'v_hZ5hgZBmUzw_263_283', 
'v_hZ5hgZBmUzw_300_316', 'v_hZ5hgZBmUzw_335_348', 'v_-ju2ucX31mY_2_17'   , 'v_-ju2ucX31mY_17_29'  , 'v_-ju2ucX31mY_34_50'  , 'v_-ju2ucX31mY_50_61'  , 
'v_-ju2ucX31mY_63_73'  , 'v_-ju2ucX31mY_113_128', 'v_-ju2ucX31mY_128_144', 'v_-ju2ucX31mY_144_161', 'v_-ju2ucX31mY_161_176', 'v_-ju2ucX31mY_176_194', 
'v_-ju2ucX31mY_194_218', 'v_-ju2ucX31mY_235_258', 'v_-ju2ucX31mY_277_296', 'v_-ju2ucX31mY_296_320', 'v_-ju2ucX31mY_334_347', 'v_-ju2ucX31mY_365_376', 
'v_-ju2ucX31mY_422_438', 'v_-ju2ucX31mY_456_467', 'v_-ju2ucX31mY_473_499', 'v_-ju2ucX31mY_499_515', 'v_Bkb6qU220QY_51_69'  , 'v_Bkb6qU220QY_95_119' , 
'v_Bkb6qU220QY_119_127', 'v_Bkb6qU220QY_128_148', 'v_Bkb6qU220QY_182_205', 'v_Bkb6qU220QY_205_255', 'v_Bkb6qU220QY_256_276', 'v_Bkb6qU220QY_318_330', 
'v_Bkb6qU220QY_331_345', 'v_Bkb6qU220QY_378_393', 'v_Bkb6qU220QY_394_409', 'v_Bkb6qU220QY_409_440', 'v_Bkb6qU220QY_458_481', 'v_Bkb6qU220QY_481_498', 
'v_Bkb6qU220QY_499_516', 'v_q7t3WICNpb4_18_30'  , 'v_q7t3WICNpb4_42_59'  , 'v_q7t3WICNpb4_97_105' , 'v_q7t3WICNpb4_105_114', 'v_q7t3WICNpb4_115_125', 
'v_q7t3WICNpb4_125_151', 'v_q7t3WICNpb4_151_169', 'v_q7t3WICNpb4_169_196', 'v_q7t3WICNpb4_196_208', 'v_q7t3WICNpb4_208_218', 'v_q7t3WICNpb4_254_261', 
'v_q7t3WICNpb4_261_269', 'v_q7t3WICNpb4_270_279', 'v_q7t3WICNpb4_279_292', 'v_q7t3WICNpb4_292_299', 'v_q7t3WICNpb4_300_308', 'v_q7t3WICNpb4_308_318', 
'v_q7t3WICNpb4_330_338', 'v_q7t3WICNpb4_338_352', 'v_q7t3WICNpb4_352_361', 'v_q7t3WICNpb4_361_376', 'v_q7t3WICNpb4_376_388', 'v_q7t3WICNpb4_388_398', 
'v_q7t3WICNpb4_398_409', 'v_q7t3WICNpb4_409_417', 'v_q7t3WICNpb4_417_429', 'v_q7t3WICNpb4_448_467', 'v_q7t3WICNpb4_467_479', 'v_q7t3WICNpb4_495_501', 
'v_q7t3WICNpb4_501_509', 'v_q7t3WICNpb4_509_522', 'v_q7t3WICNpb4_531_545', 'v_q7t3WICNpb4_545_552', 'v_q7t3WICNpb4_552_560', 'v_q7t3WICNpb4_568_579', 
'v_q7t3WICNpb4_579_592', 'v_q7t3WICNpb4_592_610', 'v_Zp0zqugqZ20_3_14'   , 'v_Zp0zqugqZ20_14_31'  , 'v_Zp0zqugqZ20_31_46'  , 'v_Zp0zqugqZ20_46_69'  , 
'v_Zp0zqugqZ20_69_84'  , 'v_Zp0zqugqZ20_101_108', 'v_Zp0zqugqZ20_108_116', 'v_Zp0zqugqZ20_125_141', 'v_Zp0zqugqZ20_141_158', 'v_Zp0zqugqZ20_158_175', 
'v_Zp0zqugqZ20_175_187', 'v_Zp0zqugqZ20_188_199', 'v_Zp0zqugqZ20_199_217', 'v_Zp0zqugqZ20_217_231', 'v_5rEGESxFj4Q_2_17'   , 'v_5rEGESxFj4Q_17_33'  , 
'v_5rEGESxFj4Q_33_51'  , 'v_5rEGESxFj4Q_51_64'  , 'v_5rEGESxFj4Q_64_90'  , 'v_5rEGESxFj4Q_91_102' , 'v_5rEGESxFj4Q_102_120', 'v_5rEGESxFj4Q_120_140', 
'v_5rEGESxFj4Q_164_185', 'v_5rEGESxFj4Q_218_240', 'v_5rEGESxFj4Q_414_431', 'v_aD1S8gDBMFE_3_18'   , 'v_aD1S8gDBMFE_18_44'  , 'v_aD1S8gDBMFE_44_55'  , 
'v_aD1S8gDBMFE_55_68'  , 'v_aD1S8gDBMFE_69_79'  , 'v_aD1S8gDBMFE_107_133', 'v_aD1S8gDBMFE_133_153', 'v_aD1S8gDBMFE_163_196', 'v_aD1S8gDBMFE_196_220', 
'v_aD1S8gDBMFE_220_241', 'v_aD1S8gDBMFE_241_269', 'v_aD1S8gDBMFE_269_284', 'v_aD1S8gDBMFE_324_355', 'v_aD1S8gDBMFE_355_363', 'v_aD1S8gDBMFE_363_373', 
'v_aD1S8gDBMFE_373_393', 'v_aD1S8gDBMFE_418_426', 'v_aD1S8gDBMFE_426_443', 'v_aD1S8gDBMFE_443_456', 'v_aD1S8gDBMFE_456_496', 'v_aD1S8gDBMFE_496_520', 
'v_aD1S8gDBMFE_520_544', 'v_aD1S8gDBMFE_545_570']

cliped_root = 'clipped_videos'
copy2root = 'test_videos'
for video_name in video_names:
    if os.path.exists(os.path.join(cliped_root, video_name[2:]+'.mp4')): 
        shutil.copyfile(os.path.join(cliped_root, video_name[2:]+'.mp4'), os.path.join(copy2root, video_name[2:]+'.mp4'))
    else:
        print(f'{video_name} not exist')
