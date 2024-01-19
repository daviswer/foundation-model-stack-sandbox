import pytest
import torch

from fms.models.gpt_bigcode import GPTBigCode, GPTBigCodeConfig, GPTBigCodeHeadless
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)


class GPTBigCodeFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base GPT-BigCode Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: GPTBigCodeConfig):
        return GPTBigCode(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> GPTBigCodeConfig:
        return GPTBigCodeConfig(
            src_vocab_size=384,
            emb_dim=16,
            nheads=8,
            nlayers=2,
            max_pos=512,
            hidden_grow_factor=2.0,
            pad_id=0,
        )


class TestGPTBigCode(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    GPTBigCodeFixtures,
):
    """
    Model Test Suite for GPT-BigCode

    This suite will include tests for:
    - model configuration
    - basic load/save model
    - consistency of model output
    """

    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]

    def test_config_passed_to_model_and_updated(self, model, config):
        """test model constructor appropriately merges any passed kwargs into the config without mutating the original config"""
        model = type(model)(config=config, pad_id=config.pad_id + 1)
        # check not same reference
        assert model.get_config() is not config

        # modify pad_id to the new value expected and check equivalence
        config.pad_id = config.pad_id + 1
        assert model.get_config().as_dict() == config.as_dict()

    @pytest.fixture
    def headless_model(self, model: GPTBigCode) -> GPTBigCodeHeadless:
        return model.base_model
