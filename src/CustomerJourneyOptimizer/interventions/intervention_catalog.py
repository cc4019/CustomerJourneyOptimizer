from typing import Dict, List

class InterventionCatalog:
    def __init__(self):
        self.interventions: Dict[str, Dict] = {}

    def add_intervention(self, intervention_id: str, name: str, description: str, category: str):
        """
        Add a new intervention to the catalog.
        
        :param intervention_id: Unique identifier for the intervention
        :param name: Name of the intervention
        :param description: Description of the intervention
        :param category: Category of the intervention
        """
        self.interventions[intervention_id] = {
            'name': name,
            'description': description,
            'category': category
        }

    def get_intervention(self, intervention_id: str) -> Dict:
        """
        Retrieve an intervention from the catalog.
        
        :param intervention_id: Unique identifier for the intervention
        :return: Dictionary containing intervention details
        """
        return self.interventions.get(intervention_id)

    def list_interventions(self) -> List[Dict]:
        """
        List all interventions in the catalog.
        
        :return: List of all interventions
        """
        return list(self.interventions.values())

    def remove_intervention(self, intervention_id: str):
        """
        Remove an intervention from the catalog.
        
        :param intervention_id: Unique identifier for the intervention
        """
        if intervention_id in self.interventions:
            del self.interventions[intervention_id]
        else:
            print(f"Intervention with ID {intervention_id} not found in the catalog.")

    def update_intervention(self, intervention_id: str, **kwargs):
        """
        Update an existing intervention in the catalog.
        
        :param intervention_id: Unique identifier for the intervention
        :param kwargs: Fields to update (name, description, category)
        """
        if intervention_id in self.interventions:
            self.interventions[intervention_id].update(kwargs)
        else:
            print(f"Intervention with ID {intervention_id} not found in the catalog.")
