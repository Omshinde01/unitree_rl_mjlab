import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

XML_FILE = r"D:\AI\Projects\Humanoid\MAIN_Training\Final_RL_training\unitree_rl_mjlab\src\assets\robots\skandharobot\xmls\SkandhaRobot.xml"

OUTPUT_DIR = Path("mujoco_parsed")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def vec_to_str(value):
    """Convert MuJoCo vector attributes to readable strings."""
    if value is None:
        return ""
    return value.strip()


def get_parent_chain(body):
    """
    Returns the body hierarchy from root -> current body.
    """
    chain = []

    while body is not None:
        chain.append(body.get("name", ""))
        body = body_parent_map.get(body)

    chain.reverse()
    return chain


# ============================================================
# LOAD XML
# ============================================================

tree = ET.parse(XML_FILE)
root = tree.getroot()

model_name = root.get("model", "")

print(f"\nMuJoCo model: {model_name}")
print("=" * 80)


# ============================================================
# BUILD BODY -> PARENT MAP
# ============================================================

body_parent_map = {}

worldbody = root.find("worldbody")

def register_bodies(parent_body=None):
    """
    Recursively register body parent relationships.
    """

    if parent_body is None:
        bodies = worldbody.findall("body")
    else:
        bodies = parent_body.findall("body")

    for body in bodies:

        body_parent_map[body] = parent_body

        register_bodies(body)


register_bodies()


# ============================================================
# 1. BODY / LINK TABLE
# ============================================================

body_rows = []

for body in body_parent_map:

    name = body.get("name", "")
    parent = body_parent_map[body]

    parent_name = (
        parent.get("name", "")
        if parent is not None
        else "worldbody"
    )

    chain = get_parent_chain(body)

    # Inertial information
    inertial = body.find("inertial")

    if inertial is not None:

        mass = inertial.get("mass", "")

        inertial_pos = inertial.get("pos", "")
        inertial_quat = inertial.get("quat", "")
        diagonal_inertia = inertial.get("diaginertia", "")

    else:

        mass = ""
        inertial_pos = ""
        inertial_quat = ""
        diagonal_inertia = ""

    body_rows.append({

        "Body / Link": name,

        "Parent Body": parent_name,

        "Depth": len(chain) - 1,

        "Hierarchy":
            " -> ".join(chain),

        "Body Position":
            body.get("pos", ""),

        "Body Quaternion":
            body.get("quat", ""),

        "Mass (kg)": mass,

        "Inertial Position":
            inertial_pos,

        "Inertial Quaternion":
            inertial_quat,

        "Diagonal Inertia":
            diagonal_inertia,

    })


body_df = pd.DataFrame(body_rows)


# ============================================================
# 2. JOINT TABLE
# ============================================================

joint_rows = []


def parse_joints(body):

    body_name = body.get("name", "")

    parent_body = body_parent_map.get(body)

    parent_name = (
        parent_body.get("name", "")
        if parent_body is not None
        else "worldbody"
    )

    chain = get_parent_chain(body)

    joints = body.findall("joint")

    for joint in joints:

        joint_type = joint.get("type", "hinge")

        range_value = joint.get("range", "")

        # Split range into min/max
        if range_value:

            range_parts = range_value.split()

            if len(range_parts) >= 2:

                range_min = range_parts[0]
                range_max = range_parts[1]

            else:

                range_min = ""
                range_max = ""

        else:

            range_min = ""
            range_max = ""

        joint_rows.append({

            # Identity
            "Joint": joint.get("name", ""),

            "Joint Type": joint_type,

            # Relationship
            "Parent Link": parent_name,

            "Child Link": body_name,

            "Hierarchy":
                " -> ".join(chain),

            # Joint geometry
            "Position":
                joint.get("pos", ""),

            "Axis":
                joint.get("axis", ""),

            # Limits
            "Range Min (rad)":
                range_min,

            "Range Max (rad)":
                range_max,

            "Range":
                range_value,

            # Actuator
            "Actuator Force Min":
                (
                    joint.get("actuatorfrcrange", "").split()[0]
                    if joint.get("actuatorfrcrange")
                    and len(joint.get("actuatorfrcrange").split()) >= 2
                    else ""
                ),

            "Actuator Force Max":
                (
                    joint.get("actuatorfrcrange", "").split()[1]
                    if joint.get("actuatorfrcrange")
                    and len(joint.get("actuatorfrcrange").split()) >= 2
                    else ""
                ),

            # Dynamics
            "Armature":
                joint.get("armature", ""),

            "Damping":
                joint.get("damping", ""),

            "Friction Loss":
                joint.get("frictionloss", ""),

        })

    # Recursively process child bodies
    for child in body.findall("body"):
        parse_joints(child)


# Start parsing
for body in worldbody.findall("body"):
    parse_joints(body)


joint_df = pd.DataFrame(joint_rows)


# ============================================================
# 3. GEOM TABLE
# ============================================================

geom_rows = []


def parse_geoms(body):

    body_name = body.get("name", "")

    parent_body = body_parent_map.get(body)

    parent_name = (
        parent_body.get("name", "")
        if parent_body is not None
        else "worldbody"
    )

    for geom in body.findall("geom"):

        geom_rows.append({

            "Geom": geom.get("name", ""),

            "Body / Link": body_name,

            "Parent Body": parent_name,

            "Type": geom.get("type", ""),

            "Mesh": geom.get("mesh", ""),

            "Position": geom.get("pos", ""),

            "Quaternion": geom.get("quat", ""),

            "Group": geom.get("group", ""),

            "Contype": geom.get("contype", ""),

            "Conaffinity": geom.get("conaffinity", ""),

            "Density": geom.get("density", ""),

            "RGBA": geom.get("rgba", ""),

        })

    for child in body.findall("body"):
        parse_geoms(child)


for body in worldbody.findall("body"):
    parse_geoms(body)


geom_df = pd.DataFrame(geom_rows)


# ============================================================
# 4. MESH / ASSET TABLE
# ============================================================

mesh_rows = []

asset = root.find("asset")

if asset is not None:

    for mesh in asset.findall("mesh"):

        mesh_rows.append({

            "Mesh Name":
                mesh.get("name", ""),

            "Mesh File":
                mesh.get("file", ""),

            "Content Type":
                mesh.get("content_type", ""),

        })


mesh_df = pd.DataFrame(mesh_rows)


# ============================================================
# 5. SITE TABLE
# ============================================================

site_rows = []


def parse_sites(body):

    body_name = body.get("name", "")

    for site in body.findall("site"):

        site_rows.append({

            "Site":
                site.get("name", ""),

            "Body / Link":
                body_name,

            "Position":
                site.get("pos", ""),

            "Quaternion":
                site.get("quat", ""),

            "Size":
                site.get("size", ""),

            "Type":
                site.get("type", ""),

            "RGBA":
                site.get("rgba", ""),

        })

    for child in body.findall("body"):
        parse_sites(child)


for body in worldbody.findall("body"):
    parse_sites(body)


site_df = pd.DataFrame(site_rows)


# ============================================================
# 6. SUMMARY
# ============================================================

summary_df = pd.DataFrame([{

    "MuJoCo Model":
        model_name,

    "Number of Bodies / Links":
        len(body_df),

    "Number of Joints":
        len(joint_df),

    "Number of Geoms":
        len(geom_df),

    "Number of Meshes":
        len(mesh_df),

    "Number of Sites":
        len(site_df),

}])


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\nMODEL SUMMARY")
print("=" * 80)

print(summary_df.to_string(index=False))

print("\nJOINTS")
print("=" * 80)

print(
    joint_df[
        [
            "Joint",
            "Joint Type",
            "Parent Link",
            "Child Link",
            "Axis",
            "Range",
            "Actuator Force Min",
            "Actuator Force Max",
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE CSV
# ============================================================

body_df.to_csv(
    OUTPUT_DIR / "bodies.csv",
    index=False
)

joint_df.to_csv(
    OUTPUT_DIR / "joints.csv",
    index=False
)

geom_df.to_csv(
    OUTPUT_DIR / "geoms.csv",
    index=False
)

mesh_df.to_csv(
    OUTPUT_DIR / "meshes.csv",
    index=False
)

site_df.to_csv(
    OUTPUT_DIR / "sites.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR / "summary.csv",
    index=False
)


# ============================================================
# SAVE EXCEL
# ============================================================

excel_file = OUTPUT_DIR / "mujoco_robot_structure.xlsx"

with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:

    summary_df.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

    body_df.to_excel(
        writer,
        sheet_name="Bodies",
        index=False
    )

    joint_df.to_excel(
        writer,
        sheet_name="Joints",
        index=False
    )

    geom_df.to_excel(
        writer,
        sheet_name="Geoms",
        index=False
    )

    mesh_df.to_excel(
        writer,
        sheet_name="Meshes",
        index=False
    )

    site_df.to_excel(
        writer,
        sheet_name="Sites",
        index=False
    )


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Excel file:       {excel_file}")
print("\nGenerated:")

print("  - summary.csv")
print("  - bodies.csv")
print("  - joints.csv")
print("  - geoms.csv")
print("  - meshes.csv")
print("  - sites.csv")
print("  - mujoco_robot_structure.xlsx")