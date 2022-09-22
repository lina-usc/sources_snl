# Authors: Christian O'Reilly <christian.oreilly@gmail.com>;
# License: MIT

from setuptools import setup


if __name__ == "__main__":
    hard_dependencies = ('numpy', 'scipy', 'mne', 'nibabel')
    install_requires = list()
    with open('requirements.txt', 'r') as fid:
        for line in fid:
            req = line.strip()
            for hard_dep in hard_dependencies:
                if req.startswith(hard_dep):
                    install_requires.append(req)

    setup(
        name='sources_snl',
        version="0.0.1",
        description='Code for EEG source reconstruction at the Speech Neuroscience Lab at UofSC.',
        python_requires='>=3.5',
        author="Christian O'Reilly",
        author_email='christian.oreilly@sc.edu',
        url='https://github.com/lina-usc/sources_snl',
        packages=['sources_snl'],
        install_requires=install_requires)
