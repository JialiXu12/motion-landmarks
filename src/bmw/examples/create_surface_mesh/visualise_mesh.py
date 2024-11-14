import morphic
reload(morphic)

visualise = True
if visualise:
    from morphic import viewer
    if "fig" not in locals():
        fig = viewer.Figure()


# Load surface meshes
bm = morphic.Mesh('./new_bm_surface.mesh')
cwm = morphic.Mesh('./new_cwm_surface.mesh')
Xn1 = bm.get_nodes(group='_default')
Xn2 = cwm.get_nodes()
Xnid1 = bm.get_node_ids(group='_default')
Xnid2 = cwm.get_node_ids(group='_default')

if visualise:
    # View breast surface mesh
    Xs1, Ts1 = bm.get_surfaces(res=16)
    Xl1 = bm.get_lines(res=32)
    
    fig.plot_surfaces('Faces1', Xs1, Ts1, color=(0,1,0), opacity=0.5)
    fig.plot_points('Nodes1', Xn1, color=(1,0,1), size=5)
    fig.plot_lines('Lines1', Xl1, color=(1,1,0), size=5)
    fig.plot_text('Text1', Xnid1[0], Xnid1[1], size=5)
    
    # View chestwall surface mesh
    Xs2, Ts2 = cwm.get_surfaces(res=16)
    Xl2 = cwm.get_lines(res=32)    
    
    fig.plot_surfaces('Faces2', Xs2, Ts2, color=(1,0,0), opacity=0.5)
    fig.plot_points('Nodes2', Xn2, color=(1,0,1), size=5)
    fig.plot_lines('Lines2', Xl2, color=(1,1,0), size=5)
    fig.plot_text('Text1', Xnid2[0], Xnid2[1], size=5)


#print bm.elements.get_groups
#print bm.nodes.get_groups('')

#for element in bm.elements:
#    print element.nodes
