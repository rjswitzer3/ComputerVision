import os

EXPECTED = { \
    'img_6858__unk.jpg' : { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_6868__ones.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_6878__twos.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_6888__threes.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_6898__sixes.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_6908__fours.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_6918__fours.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_6928__fours.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_6938__fours.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_6948__fours.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_6958__fours.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_6968__fours.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_6978__fours.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_6989__fours.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_6999__blank.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':0 }, \
    'img_7009__collection.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7019__change_in_shadows.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7029__change_in_shadows.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7039__change_in_shadows.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7050__touching.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7060__fives.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7070__sixes.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7080__sixes.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7090__sixes.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7100__unk.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7110__unk.jpg' { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7120__unk.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7130__unk.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7140__unk.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7151__unk.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7161__unk.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7171__unk.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7181__unk.jpg':{ 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7191__motion_blur.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7201__unk.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7212__motion_blur.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_7222__blank_shadow_changes.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':14 } }

def verify(output,path):
    return None)


def main():
    print('Verifying Dice')
