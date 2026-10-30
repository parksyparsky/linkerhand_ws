from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'o6_realsense_teleop'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'resource'), glob('resource/*.task')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='RealSense Skeleton Tracking and Teleoperation for LinkerHand O6',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'o6_teleop_node = o6_realsense_teleop.o6_teleop_node:main',
        ],
    },
)