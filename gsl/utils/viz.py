import pyvista as pv


def pv_cannonical_gripper(T, r=1e-3):
    center = pv.Cylinder(
        center=[0, 0, 3.3 * 1e-2], direction=[0, 0, 1], height=6.6 * 1e-2, radius=r
    )
    width = pv.Cylinder(
        center=[0, 0, 6.6 * 1e-2], direction=[1, 0, 0], height=2 * 4.1 * 1e-2, radius=r
    )
    lfinger = pv.Cylinder(
        center=[4.1 * 1e-2, 0, 8.9 * 1e-2],
        direction=[0, 0, 1],
        height=4.6 * 1e-2,
        radius=r,
    )
    rfinger = pv.Cylinder(
        center=[-4.1 * 1e-2, 0, 8.9 * 1e-2],
        direction=[0, 0, 1],
        height=4.6 * 1e-2,
        radius=r,
    )

    # gripper = pv.MultiBlock()
    # gripper.append(center.transform(T))
    # gripper.append(width.transform(T))
    # gripper.append(lfinger.transform(T))
    # gripper.append(rfinger.transform(T))
    gripper = pv.PolyData()
    gripper.merge(center.transform(T), inplace=True)
    gripper.merge(width.transform(T), inplace=True)
    gripper.merge(lfinger.transform(T), inplace=True)
    gripper.merge(rfinger.transform(T), inplace=True)

    return gripper
