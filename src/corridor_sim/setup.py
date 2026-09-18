import os
from glob import glob
from setuptools import setup

package_name = 'corridor_sim'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],

    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/corridor_sim']),

        ('share/corridor_sim',
         ['package.xml']),

        ('share/corridor_sim/launch',
         glob('launch/*.py')),

        ('share/corridor_sim/urdf',
         glob('urdf/*')),

        ('share/corridor_sim/worlds',
         glob('worlds/*')),

        ('share/corridor_sim/config',
         glob('config/*')),
    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='user',
    maintainer_email='user@test.com',

    description='Degenerate corridor simulation in ROS 2 Humble',
    license='Apache-2.0',

    tests_require=['pytest'],

    entry_points={
        'console_scripts': [
            'wall_shifter = corridor_sim.wall_shifter:main',
            'scan_to_cloud = corridor_sim.scan_to_cloud:main',
            'noise_injector = corridor_sim.noise_injector:main',
            'task2_estimator = corridor_sim.estimator_node:main',
            'trajectory_recorder = corridor_sim.trajectory_recorder:main',
            'task2_fusion = corridor_sim.task2_fusion:main',
            'pointcloud_recorder = corridor_sim.pointcloud_recorder:main',
        ],
    },
)
