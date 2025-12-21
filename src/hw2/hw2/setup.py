from setuptools import find_packages, setup

package_name = 'hw2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rose5710',
    maintainer_email='rose5710@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'publisher = hw2.publisher:main',
                'subscriber = hw2.subscriber:main',
                'cubic_trajectory_generation = hw2.cubic_trajectory_generation:main',
        ],
    },
)
