import model
import data_synthetic
import data_optiondx
import plotter

test_float, rand_int, n_trees = 0.3, 1, 100

# run 'calls' or 'puts'
def run(type_:str, test_float_:float, rand_int_:int, n_trees_:int):

    # synthetic data
    df_synth, features_synth, target_synth = data_synthetic.get_data(type_)

    option_synth, predictions_synth, mse_synth  = model.random_forest_model(df_synth, features_synth, target_synth, test_float_, rand_int_, n_trees_)

    print("synthetic data set", df_synth)
    print("synthetic prediction data_set", predictions_synth)
    print("synthetic mean squared error", mse_synth)

    # option dx data
    df_odx, features_odx, target_odx = data_optiondx.get_data(type_)

    option_odx, predictions_odx, mse_odx = model.random_forest_model(df_odx, features_odx, target_odx, test_float_, rand_int_, n_trees_)
    
    print("optiondx data set", df_odx)
    print("optiondx prediction data_set", predictions_odx)
    print("optiondx mean squared error", mse_odx)

    # plotting
    if type_ == "calls":
        naming = data_optiondx.naming_convention_calls
    else:
        naming = data_optiondx.naming_convention_puts

    plotter.plot('synthetic', naming, df_synth)
    plotter.plot('optionDX', naming, df_odx)

    return option_synth, predictions_synth, option_odx, predictions_odx

option_synth_c, predictions_synth_c, option_odx_c, predictions_odx_c = run("calls", test_float, rand_int, n_trees)
option_synth_p, predictions_synth_p, option_odx_p, predictions_odx_p = run("puts", test_float, rand_int, n_trees)

