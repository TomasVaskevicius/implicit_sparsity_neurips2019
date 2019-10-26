import numpy as np


def find_optimal_lasso_values(simulation):
    glmnet_risks = simulation['lasso_performance']['l2_squared_errors'] \
        .squeeze()
    glmnet_validation_errors = \
        simulation['lasso_performance']['validation_losses'].squeeze()
    glmnet_l_infty_S = \
        simulation['lasso_performance']['l_infty_S'].squeeze()
    glmnet_l_infty_Sc = \
        simulation['lasso_performance']['l_infty_Sc'].squeeze()

    oracle_glmnet = np.apply_along_axis(np.nanmin, 1, glmnet_risks)
    oracle_glmnet_id = np.apply_along_axis(
        np.nanargmin, 1, glmnet_risks)

    oracle_l_infty_S = glmnet_l_infty_S[
        np.arange(glmnet_risks.shape[0]), oracle_glmnet_id]
    oracle_l_infty_Sc = glmnet_l_infty_Sc[
        np.arange(glmnet_risks.shape[0]), oracle_glmnet_id]

    validation_glmnet_id = np.apply_along_axis(
        np.nanargmin, 1, glmnet_validation_errors)
    validation_glmnet = glmnet_risks[
        np.arange(glmnet_risks.shape[0]), validation_glmnet_id]
    validation_l_infty_S = glmnet_l_infty_S[
        np.arange(glmnet_risks.shape[0]), validation_glmnet_id]
    validation_l_infty_Sc = glmnet_l_infty_Sc[
        np.arange(glmnet_risks.shape[0]), validation_glmnet_id]

    return {
        'oracle_risks': oracle_glmnet,
        'validation_risks': validation_glmnet,  # True risks chosen using
        # validation data.
        'validation_l_infty_S': validation_l_infty_S,
        'validation_l_infty_Sc': validation_l_infty_Sc,
        'oracle_l_infty_S': oracle_l_infty_S,
        'oracle_l_infty_Sc': oracle_l_infty_Sc,
    }


def find_optimal_gd_values(simulation):
    gd_risks = simulation['gd_performance']['l2_squared_errors'].squeeze()
    gd_validation_errors = \
        simulation['gd_performance']['validation_losses'].squeeze()
    gd_l_infty_S = \
        simulation['gd_performance']['l_infty_S'].squeeze()
    gd_l_infty_Sc = \
        simulation['gd_performance']['l_infty_Sc'].squeeze()

    oracle_gd = np.apply_along_axis(np.nanmin, 1, gd_risks)
    oracle_gd_time = np.apply_along_axis(
        np.nanargmin, 1, gd_risks)
    oracle_l_infty_S = gd_l_infty_S[
        np.arange(gd_risks.shape[0]), oracle_gd_time]
    oracle_l_infty_Sc = gd_l_infty_Sc[
        np.arange(gd_risks.shape[0]), oracle_gd_time]

    validation_gd_time = np.apply_along_axis(
        np.nanargmin, 1, gd_validation_errors)
    validation_gd = gd_risks[
        np.arange(gd_risks.shape[0]), validation_gd_time]
    validation_l_infty_S = gd_l_infty_S[
        np.arange(gd_risks.shape[0]), validation_gd_time]
    validation_l_infty_Sc = gd_l_infty_Sc[
        np.arange(gd_risks.shape[0]), validation_gd_time]

    return {
        'oracle_risks': oracle_gd,
        'validation_risks': validation_gd,  # True risks chosen using
        # the validation data.
        'validation_l_infty_S': validation_l_infty_S,
        'validation_l_infty_Sc': validation_l_infty_Sc,
        'oracle_l_infty_S': oracle_l_infty_S,
        'oracle_l_infty_Sc': oracle_l_infty_Sc,
    }
