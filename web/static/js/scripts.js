/******************
    General Functions
*******************/

function get_selected_network() {
    var network_id = 0;
    $('#networks > tbody > tr').each(function() {
        var item = $(this).find(':input');
        if (item.prop('checked'))
            network_id = item.val()
    });
    return network_id;
}


function get_selected_dataset() {
    var dataset_id = 0;
    $('#datasets > tbody > tr').each(function() {
        var item = $(this).find(':input');
        if (item.prop('checked'))
            dataset_id = item.val()
    });
    return dataset_id;
}


function deselect_network() {
    $('#networks > tbody > tr').each(function() {
        $(this).find(':input').prop('checked', false);
        $(this).css('background-color','');
    });
}


function deselect_dataset() {
    $('#datasets > tbody > tr').each(function() {
        $(this).find(':input').prop('checked', false);
        $(this).css('background-color','');
    });
}


function isnull(value, ifnull) {
    if (value == null)
        return ifnull;
    else
        return value;
}


function round(number, precision) {
    try {
        if (number == null)
            return '';
        return parseFloat(number.toFixed(precision));	
    }
    catch(err) {
        return '';
    }
}


/******************
    Tab Functions
*******************/

function openSectionTab(evt, tabName) {
    // Declare all variables
    var i, tabcontent, tablinks;

    // Get all elements with class="tabcontent" and hide them
    tabcontent = document.getElementsByClassName("tabcontentsection");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }

    // Get all elements with class="tablinks" and remove the class "active"
    tablinks = document.getElementsByClassName("tablinkssection");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    // Show the current tab, and add an "active" class to the button that opened the tab
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}


function openDatasetTab(evt, tabName) {
    // Declare all variables
    var i, tabcontent, tablinks;

    // Get all elements with class="tabcontent" and hide them
    tabcontent = document.getElementsByClassName("tabcontentdataset");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }

    // Get all elements with class="tablinks" and remove the class "active"
    tablinks = document.getElementsByClassName("tablinksdataset");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    // Show the current tab, and add an "active" class to the button that opened the tab
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}


function openNetworkTab(evt, tabName) {
    // Declare all variables
    var i, tabcontent, tablinks;

    // Get all elements with class="tabcontent" and hide them
    tabcontent = document.getElementsByClassName("tabcontentnetwork");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }

    // Get all elements with class="tablinks" and remove the class "active"
    tablinks = document.getElementsByClassName("tablinksnetwork");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    // Show the current tab, and add an "active" class to the button that opened the tab
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}


