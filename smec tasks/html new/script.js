document.addEventListener('DOMContentLoaded', function() {
    const addDispatchForm = document.getElementById('dispatchForm');
    const modal = document.getElementById('myModal');
    let dispatchArray = JSON.parse(localStorage.getItem('dispatchArray')) || [];

    addDispatchForm.addEventListener('submit', function(event) {
        event.preventDefault();
        const dispatchData = {
            destination: document.getElementById('destination').value,
            jobNo: document.getElementById('jobNo').value,
            dcNo: document.getElementById('dcNo').value,
            invoiceNo: document.getElementById('invoiceNo').value,
            client: document.getElementById('client').value,
            courier: document.getElementById('courier').value,
            transactionNo: document.getElementById('transactionNo').value,
            noOfBoxes: document.getElementById('noOfBoxes').value
        };
        dispatchArray.push(dispatchData);
        localStorage.setItem('dispatchArray', JSON.stringify(dispatchArray));
        showModal(dispatchData);
    });

    function showModal(data) {
        const modalContent = document.getElementById('modalContent');
        modalContent.innerHTML = `
            <p>Destination: ${data.destination}</p>
            <p>Job No: ${data.jobNo}</p>
            <p>Dc No: ${data.dcNo}</p>
            <p>Invoice No: ${data.invoiceNo}</p>
            <p>Client: ${data.client}</p>
            <p>Courier: ${data.courier}</p>
            <p>Transaction No: ${data.transactionNo}</p>
            <p>No of Boxes: ${data.noOfBoxes}</p>
        `;
        modal.style.display = 'block';
    }

    // Close modal when close button or outside modal is clicked
    modal.addEventListener('click', function(event) {
        if (event.target === modal || event.target.classList.contains('close')) {
            modal.style.display = 'none';
        }
    });

    // Edit button functionality
    const editBtn = document.getElementById('editBtn');

    editBtn.addEventListener('click', function() {
        modal.style.display = 'none'; // Close modal
        const index = dispatchArray.findIndex(item => item.jobNo === editBtn.dataset.jobNo);
        if (index !== -1) {
            const editedData = dispatchArray[index];
            document.getElementById('destination').value = editedData.destination;
            document.getElementById('jobNo').value = editedData.jobNo;
            document.getElementById('dcNo').value = editedData.dcNo;
            document.getElementById('invoiceNo').value = editedData.invoiceNo;
            document.getElementById('client').value = editedData.client;
            document.getElementById('courier').value = editedData.courier;
            document.getElementById('transactionNo').value = editedData.transactionNo;
            document.getElementById('noOfBoxes').value = editedData.noOfBoxes;
        }
    });

    // Delete button functionality
    const deleteBtn = document.getElementById('deleteBtn');

    deleteBtn.addEventListener('click', function() {
        modal.style.display = 'none'; // Close modal
        const index = dispatchArray.findIndex(item => item.jobNo === deleteBtn.dataset.jobNo);
        if (index !== -1) {
            dispatchArray.splice(index, 1);
            localStorage.setItem('dispatchArray', JSON.stringify(dispatchArray));
        }
    });
});
