# read in solution, which may be split into multiple files
$results_folder = "results"
$mesh_type = "p" # prone mesh

$volunteer_id = $ARGV[$i]
for ($i = 1; $i <= 2; $i++) {
    if (defined $ARGV[$i]) {
        if ($ARGV[$i] eq "s") {
            $mesh_type = "s" # supine mesh
        } elsif ($ARGV[$i] eq "u") {
            $mesh_type eq "u" # unloaded state mesh
        }
    }
}

$filename_postfix = ""
if ($mesh_type eq "u") {
    $filename_postfix = "_0"
} elsif ($mesh_type eq "s") {
    $filename_postfix = "_1"
}

$filename = sprintf("%s/VL%05d/field%s.part0.exnode",$results_folder,$volunteer_id,$filename_postfix)
gfx read node $filename
$filename = sprintf("%s/VL%05d/field%s.part0.exelem",$results_folder,$volunteer_id,$filename_postfix)
gfx read elem $filename

#gfx read node "$results_folder/VL00055/parameter_set_0_field.part0.exnode";
#gfx read elem "$results_folder/VL00055/parameter_set_0_field.part0.exelem";

# display deformed geometry
gfx define faces egroup "Region"
gfx modify g_element Region general clear circle_discretization 6 default_coordinate Geometry0 element_discretization "16*16*16" native_discretization none;
gfx modify g_element Region lines line_width 2 select_on material green selected_material default_selected;
gfx modify g_element Region surfaces select_on invisible material default selected_material default_selected render_shaded;
gfx modify g_element Region surfaces face xi3_0 select_on material muscle selected_material default_selected render_shaded;
gfx modify g_element Region surfaces face xi3_1 select_on material tissue selected_material default_selected render_shaded;
gfx modify g_element Region surfaces face xi2_0 select_on invisible material default selected_material default_selected render_shaded;
gfx modify g_element Region surfaces face xi2_1 select_on material default selected_material default_selected render_shaded;
gfx modify g_element Region surfaces face xi1_0 select_on invisible material default selected_material default_selected render_shaded;
gfx modify g_element Region surfaces face xi1_1 select_on invisible material default selected_material default_selected render_shaded;

#gfx cre egroup Elem1
#gfx mod egroup Elem1 add element 13, 14, 15, 24, 25, 26, 35, 36, 37, 46, 47, 48

gfx edit scene
gfx create window 1 double_buffer;
gfx modify window 1 image scene default light_model default;
gfx modify window 1 layout simple ortho_axes z -y eye_spacing 0.25 width 1715 height 1005;
gfx modify window 1 set current_pane 1;
gfx modify window 1 background colour 1 1 1 texture none;
gfx modify window 1 view parallel eye_point -27.893 621.703 -716.987 interest_point 267.027 229.843 158.93 up_vector -0.955696 -0.102393 0.275974 view_angle 28.5452 near_clipping_plane 10.0388 far_clipping_plane 3587.51 relative_viewport ndc_placement -1 1 2 2 viewport_coordinates 0 0 1 1;
gfx modify window 1 overlay scene none;
gfx modify window 1 set transform_tool current_pane 1 std_view_angle 40 normal_lines antialias 2 depth_of_field 0.0 fast_transparency blend_normal;



