class LearningIndex:
    TEST = 0
    TRAINING = 1
    VALIDATION = 2
    MAX_NUM_COMPONENT = 3
    INDEX_DIC = {TEST:"TEST", TRAINING:"TRAINING", VALIDATION:"VALIDATION"}

class Component:
    COMPONENT_Y = 0
    COMPONENT_Cb = 1
    COMPONENT_Cr = 2
    MAX_NUM_COMPONENT = 3
    INDEX_DIC = {COMPONENT_Y: 'Y', COMPONENT_Cb: 'Cb', COMPONENT_Cr: 'Cr'}


class ChromaFormat:
    YCbCr4_0_0 = 0
    YCbCr4_2_0 = 1
    YCbCr4_4_4 = 2
    MAX_NUM_COMPONENT = 3

class ChannelType:
    CHANNEL_TYPE_LUMA = 0
    CHANNEL_TYPE_CHROMA = 1
    MAX_NUM_CHANNEL_TYPE = 2
    STR_LUMA = 'LUMA'
    STR_CHROMA = 'CHROMA'

class PictureFormat:
    ORIGINAL = 0
    PREDICTION = 1
    RECONSTRUCTION = 2
    UNFILTEREDRECON = 3
    MAX_NUM_COMPONENT = 4
    INDEX_DIC = {ORIGINAL:'ORIGINAL', PREDICTION:'PREDICTION',
                 RECONSTRUCTION:'RECONSTRUCTION',
                 UNFILTEREDRECON:'UNFILTEREDRECON'}

class UnitFormat:
    CU = 0
    PU = 1
    TU = 2
    MAX_NUM_COMPONENT = 3
    INDEX_DIC = {CU:'CU', PU:'PU', TU:'TU'}

ChromaScale = 2


