import phaser

image_path = '/home/data/breast_project/VL/proneT1'
if 'gui' not in locals():
    gui = phaser.ProjectGUI('volunteers', image_path)
    gui.configure_traits()
else:
    gui = locals()['gui']
