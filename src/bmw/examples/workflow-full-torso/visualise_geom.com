# read in solution, which may be split into multiple files
$results_folder = "results"

gfx read node "$results_folder/VL00027/geometry.part0.exnode";
gfx read elem "$results_folder/VL00027/geometry.part0.exelem";




#gfx read node "$results_folder/VL00055/parameter_set_0_field.part0.exnode";
#gfx read elem "$results_folder/VL00055/parameter_set_0_field.part0.exelem";

gfx create window 1

# display deformed geometry
gfx define faces egroup "Region"
gfx modify g_element Region general clear circle_discretization 6 default_coordinate Geometry0 element_discretization "4*4*4" native_discretization none;
gfx modify g_element Region lines select_on material green selected_material default_selected;
gfx modify g_element Region surfaces select_on material default selected_material default_selected render_shaded;
gfx modify g_element Region surfaces face xi3_0 select_on invisible material default selected_material default_selected render_shaded;
gfx modify g_element Region surfaces face xi3_1 select_on invisible material default selected_material default_selected render_shaded;
gfx modify g_element Region surfaces face xi2_0 select_on invisible material default selected_material default_selected render_shaded;
gfx modify g_element Region surfaces face xi2_1 select_on invisible material default selected_material default_selected render_shaded;
gfx modify g_element Region surfaces face xi1_0 select_on invisible material default selected_material default_selected render_shaded;
gfx modify g_element Region surfaces face xi1_1 select_on invisible material default selected_material default_selected render_shaded;

gfx cre egroup Elem1
gfx mod egroup Elem1 add element 13, 14, 15, 24, 25, 26, 35, 36, 37, 46, 47, 48

gfx edit scene
gfx modify window 1 set antialias 2
gfx modify window 1 view parallel eye_point 20 -200 20 interest_point 20 20 20 up_vector 0 0 1 view_angle 40 near_clipping_plane 1.5 far_clipping_plane 700 relative_viewport ndc_placement -1 1 2 2 viewport_coordinates 0 0 1 1

gfx create window 1 double_buffer;
gfx modify window 1 image scene default light_model default;
gfx modify window 1 layout simple ortho_axes z -y eye_spacing 0.25 width 1475 height 975;
gfx modify window 1 set current_pane 1;
gfx modify window 1 background colour 1 1 1 texture none;
gfx modify window 1 view perspective eye_point -342.767 -178.808 -772.488 interest_point 178.195 283.591 122.416 up_vector -0.838523 0.492128 0.233856 view_angle 15.1778 near_clipping_plane 11.3405 far_clipping_plane 4052.71 relative_viewport ndc_placement -1 1 2 2 viewport_coordinates 0 0 1 1;
gfx modify window 1 overlay scene none;
gfx modify window 1 set transform_tool current_pane 1 std_view_angle 40 normal_lines antialias 2 depth_of_field 0.0 fast_transparency blend_normal;
