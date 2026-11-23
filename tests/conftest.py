from pathlib import Path

import pytest

# Two stores: store 2 opens a day later. Store 1 sells nothing on 2013-01-02 for PRODUCE,
# and has a return (negative sale) for GROCERY I on 2013-01-03.
SAMPLE_FILES: dict[str, str] = {
    "train.csv": (
        "id,date,store_nbr,item_nbr,unit_sales,onpromotion\n"
        "0,2013-01-01,1,100,5.0,\n"
        "1,2013-01-01,1,101,2.0,\n"
        "2,2013-01-01,1,200,3.5,\n"
        "3,2013-01-02,1,100,4.0,False\n"
        "4,2013-01-02,2,200,1.0,True\n"
        "5,2013-01-03,1,100,-2.0,False\n"
        "6,2013-01-03,1,200,6.0,False\n"
    ),
    "items.csv": (
        "item_nbr,family,class,perishable\n100,GROCERY I,1,0\n101,GROCERY I,1,0\n200,PRODUCE,2,1\n"
    ),
    "stores.csv": (
        "store_nbr,city,state,type,cluster\n1,Quito,Pichincha,D,13\n2,Manta,Manabi,C,3\n"
    ),
    "transactions.csv": "date,store_nbr,transactions\n2013-01-02,1,2111\n",
    "oil.csv": "date,dcoilwtico\n2013-01-01,\n2013-01-02,93.14\n",
    "holidays_events.csv": (
        "date,type,locale,locale_name,description,transferred\n"
        "2013-01-01,Holiday,National,Ecuador,Primer dia del ano,False\n"
        "2013-01-02,Holiday,Local,Manta,Fundacion de Manta,False\n"
        "2013-01-03,Holiday,National,Ecuador,Moved away,True\n"
    ),
}


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A raw data folder holding tiny valid copies of every Kaggle CSV."""
    folder = tmp_path / "raw"
    folder.mkdir()
    for name, content in SAMPLE_FILES.items():
        (folder / name).write_text(content)
    return folder
