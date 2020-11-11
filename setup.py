from distutils.core import setup

setup(
    name="raha",
    version="1.23",
    packages=["raha"],
    url="https://github.com/bigdama/raha",
    license="Apache 2.0",
    author="Mohammad Mahdavi",
    author_email="moh.mahdavi.l@gmail.com",
    description="Raha and Baran: An End-to-End Data Cleaning System",
    keywords=["Data Cleaning", "Machine Learning", "Error Detection", "Error Correction", "Data Repairing"],
    install_requires=["numpy", "pandas", "scipy", "sklearn", "matplotlib", 
    "prettytable", "mwparserfromhell", "beautifulsoup4", "py7zr"],
    include_package_data=True,
    download_url="https://github.com/BigDaMa/raha/archive/v1.23.tar.gz"
)
