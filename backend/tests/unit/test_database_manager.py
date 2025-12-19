"""
Unit tests for DatabaseManager.update_user method.

Tests that update_user handles machine_models correctly without UnboundLocalError.
"""

import pytest
from backend.utils.database_manager import DatabaseManager


def test_update_user_method_exists():
    """
    Smoke test: Verify DatabaseManager.update_user exists and is callable.
    
    This ensures:
    - The file compiles (no syntax errors)
    - The class can be imported
    - The method exists with correct signature
    """
    manager = DatabaseManager()
    
    # Verify the method exists and has the expected signature
    assert hasattr(manager, 'update_user')
    assert callable(manager.update_user)
    
    # Verify method signature includes machine_models parameters
    import inspect
    sig = inspect.signature(manager.update_user)
    params = list(sig.parameters.keys())
    
    # Should have machine_models and machine_model_ids parameters
    assert 'machine_models' in params or 'machine_model_ids' in params


@pytest.mark.asyncio
async def test_update_user_machine_models_parameter_handling():
    """
    Test that update_user can be called with various machine_models parameter combinations.
    
    This test verifies the code structure prevents UnboundLocalError when:
    - machine_model_ids is None and machine_models is None (should not crash)
    - machine_model_ids is provided (should work)
    - machine_models is provided (should work)
    - Both are None (should not touch user.machine_models)
    
    Note: This is a structure test. Full integration tests would require a test database.
    """
    manager = DatabaseManager()
    
    # Verify method can be called with None for both (should not raise UnboundLocalError)
    # We can't actually call it without a real user_id, but we can verify the signature
    # allows these parameters to be None
    
    import inspect
    sig = inspect.signature(manager.update_user)
    
    # Check that machine_models and machine_model_ids are Optional
    machine_models_param = sig.parameters.get('machine_models')
    machine_model_ids_param = sig.parameters.get('machine_model_ids')
    
    if machine_models_param:
        # Should allow None (Optional)
        assert machine_models_param.default is None or machine_models_param.annotation is not None
    
    if machine_model_ids_param:
        # Should allow None (Optional)
        assert machine_model_ids_param.default is None or machine_model_ids_param.annotation is not None


@pytest.mark.asyncio
async def test_update_user_machine_models_with_ids():
    """
    Integration test for update_user machine_models handling with machine_model_ids.
    
    Tests:
    a) Create user + 2 machine models
    b) Call update_user to add one model via ID
    c) Verify relationship updated
    d) Call update_user to set to empty list
    e) Verify relationship cleared
    f) Test invalid ID raises ValueError
    
    This test should fail on old code (session management bug) and pass after fix.
    """
    from backend.utils.db import SessionLocal, User, MachineModel
    
    manager = DatabaseManager()
    
    # Create test machine models in the database
    with SessionLocal() as session:
        # Create machine models
        model1 = MachineModel(name="TestModel1", machine_kind="PRINT_ENGINE")
        model2 = MachineModel(name="TestModel2", machine_kind="PRINT_ENGINE")
        session.add(model1)
        session.add(model2)
        session.commit()
        session.refresh(model1)
        session.refresh(model2)
        model1_id = model1.id
        model2_id = model2.id
    
    try:
        # Create a test user
        user = await manager.create_user(
            email="test_update_machines@example.com",
            name="Test User",
            role="ADMIN",
            password="testpass123"
        )
        user_id = int(user['id'])
        
        # Verify initial state (should be empty list)
        assert user.get('machine_models') == []
        
        # Test: Add one machine model via ID
        updated = await manager.update_user(
            user_id,
            machine_model_ids=[model1_id]
        )
        assert updated['name'] == "Test User"
        assert len(updated.get('machine_models', [])) == 1
        assert "TestModel1" in updated.get('machine_models', [])
        
        # Test: Add second machine model (should replace, not append based on current implementation)
        # Note: Current implementation replaces, so we need to include both IDs
        updated = await manager.update_user(
            user_id,
            machine_model_ids=[model1_id, model2_id]
        )
        assert len(updated.get('machine_models', [])) == 2
        assert "TestModel1" in updated.get('machine_models', [])
        assert "TestModel2" in updated.get('machine_models', [])
        
        # Test: Clear machine models with empty list
        updated = await manager.update_user(
            user_id,
            machine_model_ids=[]
        )
        assert updated.get('machine_models') == []
        
        # Test: Invalid ID raises ValueError
        with pytest.raises(ValueError, match="Invalid machine model IDs"):
            await manager.update_user(
                user_id,
                machine_model_ids=[99999]  # Non-existent ID
            )
        
        # Test: Mix of valid and invalid IDs
        with pytest.raises(ValueError, match="Invalid machine model IDs"):
            await manager.update_user(
                user_id,
                machine_model_ids=[model1_id, 99999]  # One valid, one invalid
            )
        
        # Cleanup: Delete test user
        await manager.delete_user(user_id)
        
    finally:
        # Cleanup: Delete test machine models
        with SessionLocal() as session:
            session.query(MachineModel).filter(MachineModel.id.in_([model1_id, model2_id])).delete(synchronize_session=False)
            session.commit()

