from setuptools import setup, find_packages

setup(name='gym_tictactoe',
      version='0.0.2',
      description='Light TicTacToe OpenAi Gym environment',
      url='https://github.com/ClementRomac/gym-tictactoe',
      author='Clement Romac',
      author_email='clement.romac@gmail.com',
      license='MIT License',
      packages=find_packages(),
      package_data={},
      zip_safe=False,
      install_requires=['gym>=0.2.3'],
      dependency_links=[]
      )
