import os

EXPECTED = { \
    'img_6858__unk.jpg' : { 1:3, 2:1, 3:1, 4:6, 5:3, 6:0, 'unknown':0, 'total_sum':0, 'count':14 }, \
    'img_6868__ones.jpg': { 1:14, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':14, 'count':14 }, \
    'img_6878__twos.jpg': { 1:0, 2:14, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':28, 'count':14 }, \
    'img_6888__threes.jpg': { 1:0, 2:0, 3:14, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':42, 'count':14 }, \
    'img_6898__sixes.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:14, 'unknown':0, 'total_sum':84, 'count':14 }, \
    'img_6908__fours.jpg': { 1:0, 2:0, 3:0, 4:14, 5:0, 6:0, 'unknown':0, 'total_sum':56, 'count':14 }, \
    'img_6918__fours.jpg': { 1:0, 2:0, 3:0, 4:14, 5:0, 6:0, 'unknown':0, 'total_sum':56, 'count':14 }, \
    'img_6928__fours.jpg': { 1:0, 2:0, 3:0, 4:14, 5:0, 6:0, 'unknown':0, 'total_sum':56, 'count':14 }, \
    'img_6938__fours.jpg': { 1:0, 2:0, 3:0, 4:14, 5:0, 6:0, 'unknown':0, 'total_sum':56, 'count':14 }, \
    'img_6948__fours.jpg': { 1:0, 2:0, 3:0, 4:11, 5:0, 6:0, 'unknown':0, 'total_sum':44, 'count':11 }, \
    'img_6958__fours.jpg': { 1:0, 2:0, 3:0, 4:9, 5:0, 6:0, 'unknown':0, 'total_sum':36, 'count':9 }, \
    'img_6968__fours.jpg': { 1:0, 2:0, 3:0, 4:3, 5:0, 6:0, 'unknown':0, 'total_sum':12, 'count':3 }, \
    'img_6978__fours.jpg': { 1:0, 2:0, 3:0, 4:2, 5:0, 6:0, 'unknown':0, 'total_sum':8, 'count':2 }, \
    'img_6989__fours.jpg': { 1:0, 2:0, 3:0, 4:1, 5:0, 6:0, 'unknown':0, 'total_sum':4, 'count':1 }, \
    'img_6999__blank.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':0 }, \
    'img_7009__collection.jpg': { 1:4, 2:1, 3:1, 4:5, 5:1, 6:2, 'unknown':0, 'total_sum':46, 'count':14 }, \
    'img_7019__change_in_shadows.jpg': { 1:2, 2:2, 3:2, 4:3, 5:1, 6:4, 'unknown':0, 'total_sum':53, 'count':14 }, \
    'img_7029__change_in_shadows.jpg': { 1:2, 2:1, 3:3, 4:4, 5:2, 6:2, 'unknown':0, 'total_sum':51, 'count':14 }, \
    'img_7039__change_in_shadows.jpg': { 1:1, 2:5, 3:2, 4:1, 5:3, 6:1, 'unknown':0, 'total_sum':42, 'count':14 }, \
    'img_7050__touching.jpg': { 1:0, 2:0, 3:0, 4:0, 5:14, 6:0, 'unknown':0, 'total_sum':70, 'count':14 }, \
    'img_7060__fives.jpg': { 1:0, 2:0, 3:0, 4:0, 5:14, 6:0, 'unknown':0, 'total_sum':70, 'count':14 }, \
    'img_7070__sixes.jpg': { 1:0, 2:0, 3:0, 4:0, 5:14, 6:0, 'unknown':0, 'total_sum':70, 'count':14 }, \
    'img_7080__sixes.jpg': { 1:0, 2:0, 3:0, 4:0, 5:14, 6:0, 'unknown':0, 'total_sum':70, 'count':14 }, \
    'img_7090__sixes.jpg': { 1:0, 2:0, 3:0, 4:0, 5:14, 6:0, 'unknown':0, 'total_sum':70, 'count':14 }, \
    'img_7100__unk.jpg': { 1:1, 2:0, 3:0, 4:0, 5:1, 6:0, 'unknown':0, 'total_sum':6, 'count':2 }, \
    'img_7110__unk.jpg' :{ 1:1, 2:1, 3:2, 4:2, 5:1, 6:1, 'unknown':0, 'total_sum':28, 'count':8 }, \
    'img_7120__unk.jpg': { 1:1, 2:2, 3:3, 4:4, 5:0, 6:0, 'unknown':0, 'total_sum':28, 'count':10 }, \
    'img_7130__unk.jpg': { 1:2, 2:1, 3:3, 4:5, 5:2, 6:1, 'unknown':0, 'total_sum':48, 'count':14 }, \
    'img_7140__unk.jpg': { 1:3, 2:2, 3:2, 4:3, 5:3, 6:1, 'unknown':0, 'total_sum':46, 'count':14 }, \
    'img_7151__unk.jpg': { 1:3, 2:1, 3:3, 4:1, 5:2, 6:0, 'unknown':0, 'total_sum':28, 'count':10 }, \
    'img_7161__unk.jpg': { 1:4, 2:2, 3:2, 4:0, 5:1, 6:1, 'unknown':0, 'total_sum':25, 'count':10 }, \
    'img_7171__unk.jpg': { 1:0, 2:2, 3:0, 4:2, 5:2, 6:4, 'unknown':0, 'total_sum':46, 'count':10 }, \
    'img_7181__unk.jpg': { 1:1, 2:0, 3:0, 4:3, 5:0, 6:1, 'unknown':0, 'total_sum':19, 'count':5 }, \
    'img_7191__motion_blur.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':5, 'total_sum':0, 'count':5 }, \
    'img_7201__unk.jpg': { 1:0, 2:1, 3:1, 4:1, 5:1, 6:1, 'unknown':0, 'total_sum':20, 'count':5 }, \
    'img_7212__motion_blur.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':5, 'total_sum':0, 'count':5 }, \
    'img_7222__blank_shadow_changes.jpg': { 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 'unknown':0, 'total_sum':0, 'count':0 } }

def verify(output,path):
    return None


def main():
    print('Verifying Dice')
