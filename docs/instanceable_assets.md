## A Note on Instanceable USD Assets

The following section presents a method that modifies existing USD assets 
which allows Isaac Sim to load significantly more environments. This is currently
an experimental method and has thus not been completely integrated into the 
framework. As a result, this section is reserved for power users who wish to 
maxmimize the performance of the Isaac Sim RL framework. 


### Motivation

One common issue in Isaac Sim that occurs when we try to increase 
the number of environments `numEnvs` is running out of RAM. This occurs because 
the Isaac Sim RL framework uses `omni.isaac.cloner` to duplicate environments. 
As a result, there are `numEnvs` number of identical copies of the visual and 
collision meshes in the scene, which consumes lots of memory. However, only one
copy of the meshes are needed on stage since prims in all other environments could 
merely reference that one copy, thus reducing the amount of memory used for loading 
environments. To enable this functionality, USD assets need to be modified to be
`instanceable`.


### Creating Instanceable Assets

Assets can now be directly imported as Instanceable assets through the URDF and MJCF importers provided in Isaac Sim. By selecting this option, imported assets will be split into two separate USD files that follow the above hierarchy definition. Any mesh data will be written to an USD stage to be referenced by the main USD stage, which contains the main robot definition. 

To use the Instanceable option in the importers, first check the `Create Instanceable Asset` option. Then, specify a file path to indicate the location for saving the mesh data in the `Instanceable USD Path` textbox. This will default to `./instanceable_meshes.usd`, which will generate a file `instanceable_meshes.usd` that is saved to the current directory.

Once the asset is imported with these options enabled, you will see the robot definition in the stage - we will refer to this stage as the master stage. If we expand the robot hierarchy in the Stage, we will notice that the parent prims that have mesh decendants have been marked as Instanceable and they reference a prim in our `Instanceable USD Path` USD file. We are also no longer able to modify attributes of descendant meshes.

To add the instanced asset into a new stage, we will simply need to add the master USD file.


### Converting Existing Assets

We provide the utility function `convert_asset_instanceable`, which creates an instanceable 
version of a given USD asset in `/omniisaacgymenvs/utils/usd_utils/create_instanceable_assets.py`. 
To run this function, launch Isaac Sim and open the script editor via `Window -> Script Editor`.
Enter the following script and press `Run (Ctrl + Enter)`:

```bash
from omniisaacgymenvs.utils.usd_utils.create_instanceable_assets import convert_asset_instanceable
convert_asset_instanceable(
    asset_usd_path=ASSET_USD_PATH, 
    source_prim_path=SOURCE_PRIM_PATH, 
    save_as_path=SAVE_AS_PATH
)
```

Note that `ASSET_USD_PATH` is the file path to the USD asset (*e.g.* robot_asset.usd). 
`SOURCE_PRIM_PATH` is the USD path of the root prim of the asset on stage. `SAVE_AS_PATH` 
is the file path of the generated instanceable version of the asset 
(*e.g.* robot_asset_instanceable.usd). 

Assuming that `SAVE_AS_PATH` is `OUTPUT_NAME.usd`, the above script will generate two files:
`OUTPUT_NAME.usd` and `OUTPUT_NAME_meshes.usd`. `OUTPUT_NAME.usd` is the instanceable version
of the asset that can be imported to stage and used by `omni.isaac.cloner` to create numerous
duplicates without consuming much memory. `OUTPUT_NAME_meshes.usd` contains all the visual
and collision meshes that `OUTPUT_NAME.usd` references.  

It is worth noting that any [USD Relationships](https://graphics.pixar.com/usd/dev/api/class_usd_relationship.html) 
on the referenced meshes are removed in `OUTPUT_NAME.usd`. This is because those USD Relationships
originally have targets set to prims in `OUTPUT_NAME_meshes.usd` and hence cannot be accessed 
from `OUTPUT_NAME.usd`. Common examples of USD Relationships that could exist on the meshes are
visual materials, physics materials, and filtered collision pairs. Therefore, it is recommanded
to set these USD Relationships on the meshes' parent Xforms instead of the meshes themselves.

In a case where we would like to update the main USD file where the instanceable USD file is being referenced from, we also provide a utility method to update all references in the stage that matches a source reference path to a new USD file path.

```bash
from omniisaacgymenvs.utils.usd_utils.create_instanceable_assets import update_reference
update_reference(
    source_prim_path=SOURCE_PRIM_PATH, 
    source_reference_path=SOURCE_REFERENCE_PATH,
    target_reference_path=TARGET_REFERENCE_PATH
)
```

### Limitations

USD requires a specific structure in the asset tree definition in order for the instanceable flag to take action. To mark any mesh or primitive geometry prim in the asset as instanceable, the mesh prim requires a parent Xform prim to be present, which will be used to add a reference to a master USD file containing definition of the mesh prim. 

For example, the following definition:

```
	World
	  |_ Robot
	       |_ Collisions
	               |_ Sphere
	               |_ Box
```

would have to be modified to:

```
	World
	  |_ Robot
	       |_ Collisions
	               |_ Sphere_Xform
	               |      |_ Sphere
	               |_ Box_Xform
	                      |_ Box
```

Any references that exist on the original `Sphere` and `Box` prims would have to be moved to `Sphere_Xform` and `Box_Xform` prims.

To help with the process of creating new parent prims, we provide a utility method `create_parent_xforms()` in `omniisaacgymenvs/utils/usd_utils/create_instanceable_assets.py` to automatically insert a new Xform prim as a parent of every mesh prim in the stage. This method can be run on an existing non-instanced USD file for an asset from the script editor:

```bash
from omniisaacgymenvs.utils.usd_utils.create_instanceable_assets import create_parent_xforms
create_parent_xforms(
    asset_usd_path=ASSET_USD_PATH, 
    source_prim_path=SOURCE_PRIM_PATH, 
    save_as_path=SAVE_AS_PATH
)
```

This method can also be run as part of `convert_asset_instanceable()` method, by passing in the argument `create_xforms=True`.

It is also worth noting that once an instanced asset is added to the stage, we can no longer modify USD attributes on the instanceable prims. For example, to modify attributes of collision meshes that are set as instanceable, we have to first modify the attributes on the corresponding prims in the master prim which our instanced asset references from. Then, we can allow the instanced asset to pick up the updated values from the master prim.