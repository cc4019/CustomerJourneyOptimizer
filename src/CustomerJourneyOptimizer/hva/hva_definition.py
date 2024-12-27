class HVADefinition:
    def __init__(self):
        self.hvas = {}

    def add_hva(self, hva_id, name, description, measurement):
        """
        Add a new High Value Action (HVA) to the definition.
        
        :param hva_id: Unique identifier for the HVA
        :param name: Name of the HVA
        :param description: Description of the HVA
        :param measurement: How the HVA is measured
        """
        self.hvas[hva_id] = {
            'name': name,
            'description': description,
            'measurement': measurement
        }

    def get_hva(self, hva_id):
        """
        Retrieve an HVA from the definition.
        
        :param hva_id: Unique identifier for the HVA
        :return: Dictionary containing HVA details
        """
        return self.hvas.get(hva_id)

    def list_hvas(self):
        """
        List all HVAs in the definition.
        
        :return: List of all HVAs
        """
        return list(self.hvas.values())

    