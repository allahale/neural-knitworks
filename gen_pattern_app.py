import streamlit as st



def main():
    st.title('Generate your own Neural Knitworks Pattern')
    '''Some background about how this model was generated and what you can 
    do with it.'''

    # Allow user to specify dimensions in the sidebar
    # Stitches are the width of the pattern
    # Rows are the height of the pattern
    st.sidebar.title('Dimensions')
    default_dimensions = [24, 50]
    
    stitches = st.sidebar.number_input('Pattern Stitches'
        , value=default_dimensions[0]
        , min_value=1
        , max_value=200
        )
    rows = st.sidebar.number_input('Pattern Rows'
        , value=default_dimensions[1]
        , min_value=1
        , max_value=200
        )

    # Load model

    # Generate pattern
    st.sidebar.write('Pressing the generate chart button below will generate a chart')
    st.sidebar.button('Generate chart')

    # Display pattern

    # Create downloadable chart
    st.sidebar.button('Download Chart')


if __name__ == "__main__":
    main()