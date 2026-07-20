import streamlit as st 
import polars as pl


def main():
    st.header("Hello, welcome to my massive-pipeline project!")

    value_1 = st.number_input("Value 1")
    value_2 = st.number_input("Value 2")

    st.write(f"The sum of {value_1} + {value_2} is {value_1 + value_2}")


    print("Hello from massive-pipeline!")


if __name__ == "__main__":
    main()
